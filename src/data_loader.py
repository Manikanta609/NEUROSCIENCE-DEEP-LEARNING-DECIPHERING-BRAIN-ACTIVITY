import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticEEGDataset(Dataset):
    """
    Generates synthetic EEG data for testing the pipeline without external datasets.
    Simulates spectral properties of 5 mental states.
    
    States:
    0: Relaxed (Dominant Alpha: 8-12 Hz)
    1: Focused (Dominant Beta: 12-30 Hz)
    2: Stressed (High Beta/Gamma: >25 Hz, higher amplitude)
    3: Drowsy (Dominant Theta: 4-8 Hz, some Delta)
    4: Anxiety (High freq noise, irregular)
    """
    def __init__(self, num_samples=1000, num_channels=4, duration_sec=1.0, sfreq=250):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.duration = duration_sec
        self.sfreq = sfreq
        self.time_points = int(duration_sec * sfreq)
        self.classes = ['Relaxed', 'Focused', 'Stressed', 'Drowsy', 'Anxiety']
        
        self.data, self.labels = self._generate_data()
        
    def _generate_signal(self, state_idx):
        t = np.linspace(0, self.duration, self.time_points)
        signal = np.random.normal(0, 0.5, size=(self.num_channels, self.time_points)) # White noise base
        
        # Add state-specific spectral components
        if state_idx == 0: # Relaxed (Alpha)
            freqs = np.random.uniform(8, 12, self.num_channels)
            for ch, f in enumerate(freqs):
                signal[ch] += 1.5 * np.sin(2 * np.pi * f * t)
                
        elif state_idx == 1: # Focused (Beta)
            freqs = np.random.uniform(13, 30, self.num_channels)
            for ch, f in enumerate(freqs):
                signal[ch] += 1.2 * np.sin(2 * np.pi * f * t)
                
        elif state_idx == 2: # Stressed (High Beta/Gamma)
            freqs1 = np.random.uniform(25, 40, self.num_channels)
            freqs2 = np.random.uniform(40, 50, self.num_channels)
            for ch in range(self.num_channels):
                signal[ch] += 1.0 * np.sin(2 * np.pi * freqs1[ch] * t)
                signal[ch] += 0.8 * np.sin(2 * np.pi * freqs2[ch] * t)
                
        elif state_idx == 3: # Drowsy (Theta/Delta)
            freqs = np.random.uniform(4, 8, self.num_channels)
            delta = np.random.uniform(1, 4, self.num_channels)
            for ch in range(self.num_channels):
                signal[ch] += 2.0 * np.sin(2 * np.pi * freqs[ch] * t)
                signal[ch] += 1.5 * np.sin(2 * np.pi * delta[ch] * t)
                
        elif state_idx == 4: # Anxiety (Irregular high freq)
            # Mix of high beta and random bursts
            freqs = np.random.uniform(20, 35, self.num_channels)
            for ch, f in enumerate(freqs):
                signal[ch] += 1.0 * np.sin(2 * np.pi * f * t)
            # Add bursts
            burst_start = np.random.randint(0, self.time_points // 2)
            signal[:, burst_start:burst_start+50] += np.random.normal(0, 2.0, size=(self.num_channels, 50))

        return signal.astype(np.float32)

    def _generate_data(self):
        data = []
        labels = []
        
        logger.info(f"Generating {self.num_samples} synthetic EEG samples...")
        
        for _ in range(self.num_samples):
            label = np.random.randint(0, 5)
            signal = self._generate_signal(label)
            data.append(signal)
            labels.append(label)
            
        return np.array(data), np.array(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

class CSVDataset(Dataset):
    """
    Dataset for loading EEG features from a CSV (e.g., emotions.csv).
    Expects many feature columns and a 'label' column.
    """
    def __init__(self, csv_path):
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Detect label column
        if 'label' in self.df.columns:
            self.label_col = 'label'
        else:
            # Fallback: assume last column
            self.label_col = self.df.columns[-1]
            
        # Encode labels if they are strings
        if self.df[self.label_col].dtype == 'object':
            self.label_map = {label: i for i, label in enumerate(self.df[self.label_col].unique())}
            self.df['label_encoded'] = self.df[self.label_col].map(self.label_map)
            self.target_col = 'label_encoded'
            print(f"Encoded labels: {self.label_map}")
        else:
            self.target_col = self.label_col
            self.label_map = None

        # Feature columns (all except label and encoded label)
        self.feature_cols = [c for c in self.df.columns if c not in [self.label_col, 'label_encoded']]
        self.num_features = len(self.feature_cols)
        self.num_samples = len(self.df)
        
        # Pre-convert to tensor to save time during training
        self.X = torch.tensor(self.df[self.feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.df[self.target_col].values, dtype=torch.long)
        
        print(f"Loaded Features: {self.num_samples} samples, {self.num_features} features")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_real_dataset(csv_path, batch_size=32):
    """
    Returns a Dataset object for the CSV.
    """
    dataset = CSVDataset(csv_path)
    return dataset
