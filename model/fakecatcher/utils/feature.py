import numpy as np
from typing import Tuple
from scipy.signal import csd
from scipy.stats import entropy
from scipy.fft import fft, fftfreq

class FeatureExtractor:
    """
    """
    def __init__(self, PPG_G, PPG_C):
        self.PPG_G = PPG_G
        self.PPG_C = PPG_C
        self.S = self.prepare_S()
        self.Dc = self.prepare_Dc()

    def prepare_S():
        pass

    def prepare_Dc():
        pass

    ## ====================================================================
    ## ====================== Signal Transformations ======================
    ## ====================================================================

    def spectral_line(self, r_xxs: np.ndarray, threshold_factor=1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the number of spectral lines and maximum spectral line magnitude for each row.
        """
        magnitudes = np.abs(r_xxs)
        thresholds = threshold_factor * np.mean(magnitudes, axis=1, keepdims=True)
        spectral_lines = magnitudes > thresholds
        num_spectral_lines = np.sum(spectral_lines, axis=1)
        max_spectral_lines = np.max(magnitudes, axis=1)
        return num_spectral_lines, max_spectral_lines

    def spectral_auto_correlation(self, signals: np.ndarray, signal_length: int) -> np.ndarray:
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
    
    def autocorrelation(self, signal: np.ndarray, max_lag: int):
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
    
    def shannon_entropy(self, signal: np.ndarray, num_windows: int):
        """
        Shannon entropy
        param:
        signal: (signal_length,)
        """
        counts, _ = np.histogram(signal, bins=num_windows, range=(0, np.max(signal)))
        probabilities = counts / np.sum(counts)

        sh_entropy = entropy(probabilities, base=2)
        return sh_entropy
    
    def narrow_pulse(self, r_xxs: np.ndarray, threshold_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the number and average magnitude of narrow pulses for each row.
        """
        magnitudes = np.abs(r_xxs)
        thresholds = threshold_factor * np.mean(magnitudes, axis=1, keepdims=True)
        pulses = magnitudes > thresholds
        num_pulses = np.sum(pulses, axis=1)
        
        avg_pulses = np.where(num_pulses > 0, np.mean(magnitudes * pulses, axis=1), 0)

        return num_pulses, avg_pulses

    ## ====================================================================
    ## ========================= Feature Creation =========================
    ## ====================================================================


    def F1(self, signal1: np.ndarray, signal2: np.ndarray, nperseg: int=300) -> np.ndarray:
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


    def F3(self, signals: np.ndarray, signal_length: int) -> np.ndarray:
        """
        Extract F3 features for each row of a 2D signal array using numpy vectorization.
        params:
        signals (np.ndarray): (num_signals, signal_length)
        returns:
        np.ndarray: Array of shape (4, num_signals) containing the 4 features for each row.
        """
        r_xxs = self.spectral_auto_correlation(signals, signal_length)
        num_pulses, avg_pulses = self.narrow_pulse(r_xxs)
        num_spectral_lines, max_spectral_lines = self.spectral_line(r_xxs)

        features = np.vstack((num_pulses, num_spectral_lines, avg_pulses, max_spectral_lines))
        return features


    def F4(self, signals: np.ndarray, window_size: int): 
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
        mean_autocorrelations = np.array([self.autocorrelation(signal, max_lag) for signal in signals])

        # 7. Shannon entropy
        sh_entropy = np.array([self.shannon_entropy(signal, num_windows) for signal in signals])
        
        features = np.column_stack((std, sdann, rmssd, sdnni, sdsd, mean_autocorrelations, sh_entropy))

        return features   