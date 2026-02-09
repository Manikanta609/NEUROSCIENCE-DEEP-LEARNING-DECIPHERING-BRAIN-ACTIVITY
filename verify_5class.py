import torch
import joblib
import pandas as pd
import numpy as np
from src.models import get_model

def verify():
    print("Loading resources...")
    scaler = joblib.load('models/scaler.pkl')
    classes = joblib.load('models/label_classes.pkl')
    print(f"Classes: {classes}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('MLP', nb_classes=len(classes), Chans=None, Samples=None, input_dim=2548)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Test 1: Positive Data -> Should be Focused
    print("\n--- Testing Positive Data (Expected: Focused) ---")
    df = pd.read_csv('test_positive.csv')
    X = df.drop(columns=['label']).values
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out = model(X_t)
        preds = torch.argmax(out, dim=1).cpu().numpy()
    
    pred_labels = [classes[p] for p in preds]
    print(f"Predictions: {pred_labels[:5]}")
    
    # Test 2: Negative Data -> Should be Stressed or Anxiety
    print("\n--- Testing Negative Data (Expected: Stressed/Anxiety) ---")
    df = pd.read_csv('test_negative.csv')
    X = df.drop(columns=['label']).values
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out = model(X_t)
        preds = torch.argmax(out, dim=1).cpu().numpy()
    
    pred_labels = [classes[p] for p in preds]
    print(f"Predictions: {pred_labels[:5]}")

if __name__ == "__main__":
    verify()
