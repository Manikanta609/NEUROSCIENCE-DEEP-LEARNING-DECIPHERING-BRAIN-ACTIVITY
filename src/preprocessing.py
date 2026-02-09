import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_data(data, lowcut, highcut, fs, order=4):
    """
    Apply bandpass filter to 2D (channels, time) or 3D (epochs, channels, time) data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y

def normalize_data(data):
    """
    Z-score normalization per channel.
    Data shape: (channels, time) or (epochs, channels, time)
    """
    # For 3D data, we normalize across the time dimension for each channel/epoch
    if data.ndim == 3:
        # data: (N, C, T)
        means = np.mean(data, axis=-1, keepdims=True)
        stds = np.std(data, axis=-1, keepdims=True)
        return (data - means) / (stds + 1e-6)
    elif data.ndim == 2:
        # data: (C, T)
        means = np.mean(data, axis=-1, keepdims=True)
        stds = np.std(data, axis=-1, keepdims=True)
        return (data - means) / (stds + 1e-6)
    else:
        scaler = StandardScaler()
        return scaler.fit_transform(data)
