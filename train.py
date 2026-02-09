import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import our modules
from src.data_loader import SyntheticEEGDataset, load_real_dataset
from src.preprocessing import filter_data, normalize_data
from src.models import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        dataset = load_real_dataset(args.data_path)
    else:
        print("Using Synthetic Dataset...")
        dataset = SyntheticEEGDataset(num_samples=2000, num_channels=args.channels, 
                                      duration_sec=1.0, sfreq=args.sfreq)

    # 2. Split Data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model Setup
    if True: # Force logic block
        # Load Dataset (for MLP feature-based)
        data_path = 'data/emotions_5class.csv'
        if not os.path.exists(data_path):
             # Fallback to older csv if new one fails, though we expect it to exist
             data_path = 'data/emotions.csv'
             
        if args.model.lower() == 'mlp' and os.path.exists(data_path):
            print(f"Loading {data_path}...")
            df = pd.read_csv(data_path)
            X = df.drop(columns=['label']).values
            y = df['label'].values
            
            # Auto encoding labels
            from sklearn.preprocessing import LabelEncoder
            import joblib
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            # Save encoder classes for app.py REUSE
            # We save it so app.py can know '0' -> 'Anxiety', etc.
            joblib.dump(le.classes_, os.path.join(args.model_dir, 'label_classes.pkl'))
            print(f"Encoded Classes: {le.classes_}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Save Scaler
            joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.pkl'))
            
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
            
            # recreate dataloaders for feature training
            from torch.utils.data import TensorDataset
            train_dataset = TensorDataset(X_train_t, y_train_t)
            val_dataset = TensorDataset(X_test_t, y_test_t)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            
            input_dim = X_train.shape[1]
            nb_classes = len(le.classes_)
            print(f"Training MLP: Input Dim={input_dim}, Classes={nb_classes}")
            
            model = get_model('mlp', nb_classes=nb_classes, Chans=None, Samples=None, input_dim=input_dim)
            model.to(device)
    else:
        # Time-series data (Channels, Time)
        print(f"Detected time-series dataset. Using {args.model} model.")
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
            
        # Calculate samples per epoch based on duration and sfreq
        samples = int(1.0 * args.sfreq) # Fixed duration of 1s for synthetic
        model = get_model(args.model, nb_classes=5, Chans=args.channels, Samples=samples).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    best_acc = 0.0
    train_losses, val_losses = [], []
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print("  -> Saved best model")
            print("  -> Saved best model")
            
    # Save Scaler if it exists
    if 'scaler' in locals():
        import joblib
        scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
            
    print("Training Complete.")
    
    # 5. Final Evaluation on Validation Set
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    _, _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print("\nClassification Report:")
    # Define class names based on number of classes
    if nb_classes == 5:
        target_names = ['Relaxed', 'Focused', 'Stressed', 'Drowsy', 'Anxiety']
    elif nb_classes == 3:
        target_names = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    else:
        target_names = [f'Class {i}' for i in range(nb_classes)]
        
    try:
        print(classification_report(val_labels, val_preds, target_names=target_names))
    except ValueError as e:
        print(f"Could not generate detailed report with names: {e}")
        print(classification_report(val_labels, val_preds))
    
    # Save Loss Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.model_dir, 'loss_curve.png'))
    print(f"Loss curve saved to {args.model_dir}/loss_curve.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_synthetic', action='store_true', default=True, help='Use synthetic data')
    parser.add_argument('--data_path', type=str, default=None, help='Path to real dataset CSV')
    parser.add_argument('--model', type=str, default='eegnet', choices=['eegnet', 'lstm', 'mlp'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--sfreq', type=int, default=250)
    parser.add_argument('--model_dir', type=str, default='models')
    
    args = parser.parse_args()
    main(args)
