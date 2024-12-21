import numpy as np
from scipy.stats import entropy

def autocorrelation(signal: np.ndarray, max_lag: int):
    """
    param:
    signal (np.ndarray): shape: (signal_length,)
    max_lag (int): signal_length
    """
    # Normalization
    normalized_signal = signal - np.mean(signal)
    
    # Compute autocorrelation
    full_autocorrelation = np.correlate(normalized_signal, normalized_signal, mode='full') / (np.var(signal) * len(signal))
    
    # Extract positive lags
    autocorrelations = full_autocorrelation[len(signal) - 1:len(signal) - 1 + max_lag]
    
    # Compute mean of autocorrelation
    return np.mean(autocorrelations)

def shannon_entropy(signal: np.ndarray, num_windows: int):
    """
    Shannon entropy
    param:
    signal: (signal_length,)
    """
    counts, _ = np.histogram(signal, bins=num_windows, range=(0, np.max(signal)))
    probabilities = counts / np.sum(counts)

    sh_entropy = entropy(probabilities, base=2)
    return sh_entropy

def F4(signals: np.ndarray, window_size: int): 
    """
    Calculate 7 features for each row of a 2D signal array using numpy.
    param:
    signal (np.ndarray): Input signal of shape (num_signals, signal_length).
    window_size (int): Size of the window for analysis.

    returns:
    np.ndarray: Array of shape (num_signals, 7) containing the 7 features for each row.
    """
    num_signals, signal_length = signal.shape

    num_windows = signal_length//window_size    
    reshaped_signal = signal[:, :num_windows * window_size].reshape(num_signals, num_windows, window_size)
    
    # 1. std
    std = np.std(signal, axis=1)

    # 2. sdann
    # Reshape signal to (num_windows, window_size) and truncate extra values
    window_means = reshaped_signal.mean(axis=2)
    sdann = np.std(window_means, axis=1)

    # 3.rmssd(1sec difference)
    # 1초 간격의 signal끼리 차분
    successive_difference =  np.diff(reshaped_signal[:,:,0], axis=1)
    # 제곱의 평균의 루트 계산
    rmssd = np.sqrt(np.mean(successive_difference**2,axis=1))

    # 4. sdnni
    window_std = reshaped_signal.std(axis=2)
    sdnni = np.mean(window_std, axis=1)

    # 5. sdsd(standard deviation of differences)
    successive_differences = np.diff(signal, axis=1)
    sdsd = np.std(successive_differences, axis=1)

    # 6. mean of autocorrelation
    max_lag = signal_length
    mean_autocorrelations = np.array([autocorrelation(signal, max_lag) for signal in signals])

    # 7. Shannon entropy
    sh_entropy = np.array([shannon_entropy(signal, num_windows) for signal in signals])
    
    features = np.column_stack((std, sdann, rmssd, sdnni, sdsd, mean_autocorrelations, sh_entropy))

    return features    


if __name__ == "__main__":
    # Example signal
    signal1 = np.random.randn(1000)  # Simulated signal of length 1000
    signal2 = np.random.randn(1000)
    fps = 50  # Example window size (frames per second)
    signal = np.array([signal1, signal2])
    # Calculate RMSSD
    print(F4(signal, fps))

