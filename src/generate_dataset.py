import numpy as np
import pandas as pd
import os

def generate_csv_dataset(output_path='data/synthetic_emotions_mimic.csv', reference_file='data/emotions.csv', num_samples=200):
    print(f"Generating synthetic data mimicking {reference_file}...")
    
    if not os.path.exists(reference_file):
        print(f"Error: {reference_file} not found. Cannot mimic schema.")
        return

    # Read just the header to get column names
    try:
        # Optimization: Read only header
        with open(reference_file, 'r') as f:
            header_line = f.readline().strip()
        
        columns = header_line.split(',')
        
        # Identify label column (usually the last one)
        label_col = 'label'
        feature_cols = [c for c in columns if c != label_col]
        
        print(f"Detected {len(feature_cols)} feature columns.")
        
        # Generate random feature data
        # Using simple gaussian noise just to fill the CSV
        data = np.random.randn(num_samples, len(feature_cols))
        
        # Generate random labels (NEGATIVE, NEUTRAL, POSITIVE) based on what's typical in this dataset
        labels = np.random.choice(['NEGATIVE', 'NEUTRAL', 'POSITIVE'], size=num_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_cols)
        df[label_col] = labels
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved synthetic feature dataset to {output_path}")
        print(f"Shape: {df.shape}")
        
    except Exception as e:
        print(f"Failed to generate dataset: {e}")

if __name__ == '__main__':
    generate_csv_dataset()
