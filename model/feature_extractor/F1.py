import numpy as np
from scipy.signal import csd

def F1(signal1: np.ndarray, signal2: np.ndarray, nperseg: int=300) -> np.ndarray:
    """
    Calculate F1 features: Mean and max cross spectral density differences.

    Parameters:
    signal1 (np.ndarray): (num_signal, each_signal_length)  original ppg transformed signals
    signal2 (np.ndarray): (num_signal, each_signal_length)  synthetic ppg transformed signals .

    Returns:
    np.ndarray: Array of shape (2, ) for the 2 features in F1.
    """
    assert signal1.shape == signal2.shape, "Both signals must have the same shape."
    
    n_signals = signal1.shape[0]
    differences = []

    for i in range(n_signals):
        for j in range(n_signals):
            
            # Compute cross spectral densities
            _, csd1 = csd(signal1[i], signal1[j], nperseg=nperseg)
            _, csd2 = csd(signal2[i], signal2[j], nperseg=nperseg)

            # Compute absolute difference
            difference = np.abs(csd1 - csd2)
            differences.append(difference)

    # Calculate mean and max differences
    mean_difference = np.mean(differences)
    max_difference = np.max(differences)

    # Return as a feature vector
    return np.array([mean_difference, max_difference])

# Example usage
if __name__ == "__main__":
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)  # 1초 길이의 시간축

    # Original signal set as a 2D numpy array
    original_signal = np.array([
        np.sin(2 * np.pi * 50 * t) + 1,    # 50Hz signal
        np.sin(2 * np.pi * 100 * t) + 1,   # 100Hz signal
        np.sin(2 * np.pi * 150 * t) + 1    # 150Hz signal
    ])
    # Synthetic signal set as a 2D numpy array
    synthetic_signal = np.array([
        np.sin(2 * np.pi * 50 * t + 0.5) + 1,    # Phase-shifted 50Hz signal
        np.sin(2 * np.pi * 100 * t + 0.5) + 1,   # Phase-shifted 100Hz signal
        np.sin(2 * np.pi * 150 * t + 0.5) + 1    # Phase-shifted 150Hz signal
    ])

    # Log scale transformation
    log_o = np.log(np.abs(original_signal) + 1e-9)
    log_s = np.log(np.abs(synthetic_signal) + 1e-9)

    # Execute F1 function
    feature_vector = F1(log_o, log_s)

    # Print results
    print(f"Feature Vector (Mean, Max): {feature_vector}")
    print(f"Shape of Feature Vector: {feature_vector.shape}")
