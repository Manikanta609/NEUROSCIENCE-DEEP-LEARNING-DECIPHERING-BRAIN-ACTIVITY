import numpy as np
from scipy.signal import welch

def compute_psd_features(data, fs=250, nperseg=None):
    """
    Compute relative power in standard EEG bands.
    Bands: Delta(1-4), Theta(4-8), Alpha(8-12), Beta(12-30), Gamma(30-50)
    
    Args:
        data: (N_epochs, N_channels, N_timepoints) or (N_channels, N_timepoints)
        fs: Sampling frequency
        
    Returns:
        features: (N_epochs, N_channels * 5) - flat feature vector per epoch
    """
    if nperseg is None:
        nperseg = min(fs, data.shape[-1])
        
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=-1)
    
    # Define bands
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 50)
    }
    
    # Calculate band powers
    band_powers = []
    
    # Handle single epoch case
    if data.ndim == 2:
        # (C, T) -> (1, C, T) for uniform processing
        psd = psd[np.newaxis, ...]
        
    # psd shape is now (N, C, Freqs)
    
    all_feats = []
    
    for epoch_idx in range(psd.shape[0]):
        epoch_feats = []
        for ch_idx in range(psd.shape[1]):
            ch_psd = psd[epoch_idx, ch_idx, :]
            total_power = np.sum(ch_psd)
            
            ch_band_powers = []
            for band, (f_min, f_max) in bands.items():
                idx = np.logical_and(freqs >= f_min, freqs <= f_max)
                power = np.sum(ch_psd[idx])
                rel_power = power / (total_power + 1e-6) # Relative power
                ch_band_powers.append(rel_power)
            
            epoch_feats.extend(ch_band_powers)
        all_feats.append(epoch_feats)
        
    return np.array(all_feats) # Shape: (N_epochs, N_channels * 5)
