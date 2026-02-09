import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split

from src.data_loader import SyntheticEEGDataset, load_real_dataset
from src.models import get_model

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data (Same logic as train.py to ensure consistent split)
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        dataset = load_real_dataset(args.data_path)
    else:
        print("Using Synthetic Dataset...")
        dataset = SyntheticEEGDataset(num_samples=2000, num_channels=4)

    # 2. Split Data (Replicate Split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Detect Model Type from Data
    sample_data, _ = val_dataset[0]
    input_shape = sample_data.shape
    
    # Auto-detect classes
    all_y = [y for _, y in val_dataset]
    nb_classes = len(torch.unique(torch.tensor(all_y)))
    print(f"Detected {nb_classes} classes.")
    
    if len(input_shape) == 1:
        print("Detected feature-based input. Using MLP.")
        model_name = 'mlp'
        num_features = input_shape[0]
        model = get_model('mlp', nb_classes=nb_classes, Chans=None, Samples=None, input_dim=num_features).to(device)
    else:
        print("Detected signal input. Using EEGNet (or specified model).")
        model_name = args.model
        samples = int(1.0 * args.sfreq)
        model = get_model(model_name, nb_classes=nb_classes, Chans=input_shape[0], Samples=samples).to(device)
        
    # 4. Load Weights
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    # --- NEW: Load and Apply Scaler ---
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path) and model_name == 'mlp':
        import joblib
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        
        # Apply scaling to validation dataset features
        # Assuming val_dataset is a Subset or Dataset
        # We need to access the underlying data.
        # Ideally, we should perform transform on the fly in the Dataset class, 
        # but for consistency with train.py hack, we modify the tensor in place.
        
        # Access indices of the validation subset
        val_indices = val_dataset.indices
        
        # Get raw data
        # Check if X is explicitly available (CSVDataset stores it)
        if hasattr(val_dataset.dataset, 'X'):
            raw_X = val_dataset.dataset.X[val_indices].numpy()
            scaled_X = scaler.transform(raw_X)
            val_dataset.dataset.X[val_indices] = torch.tensor(scaled_X, dtype=torch.float32)
            print("Validation data scaled successfully.")
        else:
            print("Warning: Could not access underlying data to scale. Accuracy may be poor.")
    # ----------------------------------

    model.eval()
    
    # 5. Inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 6. Metrics & Visualization
    if nb_classes == 5:
        target_names = ['Relaxed', 'Focused', 'Stressed', 'Drowsy', 'Anxiety']
    elif nb_classes == 3:
        target_names = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    else:
        target_names = [f'Class {i}' for i in range(nb_classes)]
        
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    out_file = os.path.join(args.model_dir, 'confusion_matrix.png')
    plt.savefig(out_file)
    print(f"Confusion matrix saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='eegnet')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sfreq', type=int, default=250)
    
    args = parser.parse_args()
    evaluate(args)
