import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet implementation based on 'EEGNet: A Compact Neural Network for EEG-based
    Brain-Computer Interfaces' (Lawhern et al., 2018).
    
    Expected Input Shape: (Batch, 1, Channels, TimeSamples)
    """
    def __init__(self, nb_classes=5, Chans=4, Samples=250, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
        super(EEGNet, self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1: Temporal Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)
        )
        
        # Block 1: Depthwise Convolution (Spatial Filter)
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 2: Separable Convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 16 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (Samples // 32), nb_classes)
        )

    def forward(self, x):
        # Ensure input is 4D: (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

class SimpleLSTM(nn.Module):
    """
    A simple LSTM baseline for time-series classification.
    Input Shape: (Batch, Channels, TimeSamples) -> permuted to (Batch, Time, Channels)
    """
    def __init__(self, nb_classes=5, Chans=4, hidden_size=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=Chans, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, nb_classes)
        
    def forward(self, x):
        # x: (Batch, Channels, Time) -> (Batch, Time, Channels)
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class FeatureMLP(nn.Module):
    """
    A unified MLP for classification from flat feature vectors.
    Includes Batch Normalization and Dropout for better convergence.
    """
    def __init__(self, nb_classes=3, input_dim=2548, hidden_dim=128):
        super(FeatureMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, nb_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def get_model(model_name, nb_classes, Chans, Samples, input_dim=None):
    if model_name.lower() == 'eegnet':
        return EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    elif model_name.lower() == 'lstm':
        return SimpleLSTM(nb_classes=nb_classes, Chans=Chans)
    elif model_name.lower() == 'mlp':
        if input_dim is None:
            raise ValueError("input_dim must be provided for MLP")
        return FeatureMLP(nb_classes=nb_classes, input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
