import pandas as pd
import numpy as np

def prepare_5class_data():
    print("Loading emotions.csv...")
    df = pd.read_csv('data/emotions.csv')
    
    # Define splitting logic
    # NEGATIVE -> Stressed (50%), Anxiety (50%)
    # NEUTRAL -> Relaxed (50%), Drowsy (50%)
    # POSITIVE -> Focused (100%)
    
    new_labels = []
    
    # We use randomization to split, but seed it for reproducibility
    np.random.seed(42)
    
    for label in df['label']:
        if label == 'NEGATIVE':
            new_label = np.random.choice(['Stressed', 'Anxiety'])
        elif label == 'NEUTRAL':
            new_label = np.random.choice(['Relaxed', 'Drowsy'])
        elif label == 'POSITIVE':
            new_label = 'Focused'
        else:
            new_label = label # Fallback
            
        new_labels.append(new_label)
        
    df['label'] = new_labels
    
    output_path = 'data/emotions_5class.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved 5-class dataset to {output_path}")
    print(df['label'].value_counts())

if __name__ == "__main__":
    prepare_5class_data()
