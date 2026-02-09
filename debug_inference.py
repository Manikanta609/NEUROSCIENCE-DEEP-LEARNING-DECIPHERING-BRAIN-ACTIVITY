import torch
import joblib
import numpy as np
import pandas as pd
from src.models import get_model

def debug_inference():
    print("Loading resources...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Scaler
    scaler = joblib.load('models/scaler.pkl')
    print("Scaler loaded.")
    
    # Load Model
    model = get_model('MLP', nb_classes=3, Chans=None, Samples=None, input_dim=2548)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Load Sample Input
    try:
        with open('sample_input.txt', 'r') as f:
            content = f.read().strip()
        
        print(f"Sample input read (length: {len(content)} chars)")
        features = np.fromstring(content, sep=',')
        print(f"Parsed feature vector shape: {features.shape}")
        
        if len(features) != 2548:
            print(f"WARNING: Feature length mismatch. Expected 2548, got {len(features)}")
            
        # Scale
        scaled_features = scaler.transform([features])
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        print("\n--- Prediction Results ---")
        classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        for i, cls in enumerate(classes):
            print(f"{cls}: {probs[i]*100:.4f}%")
            
        pred_idx = np.argmax(probs)
        print(f"\nPredicted Class: {classes[pred_idx]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_inference()
