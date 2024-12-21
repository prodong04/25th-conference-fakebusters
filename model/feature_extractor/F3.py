import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from typing import List, Tuple

def spectral_auto_correlation(signals: np.ndarray, signal_length: int) -> np.ndarray:
    """
    Calculate the spectral autocorrelation for each row of a 2D signal array.
    """
    delta_f_list = np.arange(-(signal_length - 1) // 2, (signal_length + 1) // 2)
    X_f = fft(signals, axis=1)  # FFT along rows
    r_xxs = []

    for delta_f in delta_f_list:
        X_f_shifted = np.roll(X_f, shift=delta_f, axis=1)
        r_xx = np.mean(X_f * np.conjugate(X_f_shifted), axis=1)
        r_xxs.append(r_xx)

    return np.array(r_xxs).T  # Transpose to have (num_rows, len(delta_f_list))

def spectral_line(r_xxs: np.ndarray, threshold_factor=1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the number of spectral lines and maximum spectral line magnitude for each row.
    """
    magnitudes = np.abs(r_xxs)
    thresholds = threshold_factor * np.mean(magnitudes, axis=1, keepdims=True)
    spectral_lines = magnitudes > thresholds
    num_spectral_lines = np.sum(spectral_lines, axis=1)
    max_spectral_lines = np.max(magnitudes, axis=1)
    return num_spectral_lines, max_spectral_lines

def narrow_pulse(r_xxs: np.ndarray, threshold_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the number and average magnitude of narrow pulses for each row.
    """
    magnitudes = np.abs(r_xxs)
    thresholds = threshold_factor * np.mean(magnitudes, axis=1, keepdims=True)
    pulses = magnitudes > thresholds
    num_pulses = np.sum(pulses, axis=1)
    
    avg_pulses = np.where(num_pulses > 0, np.mean(magnitudes * pulses, axis=1), 0)

    return num_pulses, avg_pulses

def F3(signals: np.ndarray, signal_length: int) -> np.ndarray:
    """
    Extract F3 features for each row of a 2D signal array using numpy vectorization.
    params:
    signals (np.ndarray): (num_signals, signal_length)
    returns:
    np.ndarray: Array of shape (4, num_signals) containing the 4 features for each row.
    """
    r_xxs = spectral_auto_correlation(signals, signal_length)
    num_pulses, avg_pulses = narrow_pulse(r_xxs)
    num_spectral_lines, max_spectral_lines = spectral_line(r_xxs)

    features = np.vstack((num_pulses, num_spectral_lines, avg_pulses, max_spectral_lines))
    return features


if __name__ == "__main__":
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)  # 1초 길이의 시간축

    # Example 2D signal (3 rows, each with 1000 samples)
    signals = np.array([
        np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 100 * t),
        np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 60 * t),
        np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 40 * t)
    ])

    features = F3(signals, signals.shape[1])
    print("Features:")
    print(features)
    print("Shape of output:", features.shape)  # Should be (4, num_rows)
