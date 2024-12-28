import numpy as np
from scipy.signal import butter, filtfilt, windows
import math


class PPG_C:
    """
    G. de Haan and V. Jeanne, “Robust pulse rate from chrominance-based
    rPPG,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 10,
    pp. 2878–2886, Oct 2013.
    """
    def __init__(self, RGB_mean_array: np.ndarray, fps: int):
        """
        ROI 경로로 비디오 정보를 읽어 인스턴스 초기화.
        """
        self.RGB_mean_array = RGB_mean_array
        self.fps = fps

    ## ====================================================================
    ## ========================== Core Methods ============================
    ## ====================================================================

    def bandpass_filter(self, lpf: float, hpf: float) -> tuple:
        """
        대역 통과 필터 설계.

        Args:
            lpf: 저주파 차단 주파수.
            hpf: 고주파 차단 주파수.
        
        Returns:
            B, A: 필터 계수.
        """
        nyquist_freq = 0.5 * self.fps
        B, A = butter(3, [lpf / nyquist_freq, hpf / nyquist_freq], btype='bandpass')
        return B, A

    def compute_signal(self, lpf=0.7, hpf=2.5, win_sec=1.6, step_sec=0.7) -> np.ndarray:
        """
        CHROM 알고리즘을 사용해 PPG 신호를 계산.

        Args:
            lpf: 저주파 차단 주파수.
            hpf: 고주파 차단 주파수.
            win_sec: 슬라이딩 윈도우 길이 (초).
        
        Returns:
            BVP: PPG 신호.
        """
        # 1. RGB 평균값 추출
        num_frames = self.RGB_mean_array.shape[0]

        # 2. Band-pass filtering
        B, A = self.bandpass_filter(lpf, hpf)
        
        # 3. 슬라이딩 윈도우 설정
        win_length = int(self.fps * win_sec)
        step_size = int(self.fps * step_sec)

        # 4. PPG 신호 계산
        COMPUTED_PPG_SIGNAL = np.zeros(num_frames)  # 결과 배열을 num_frames 크기로 초기화
        win_start = 0

        while win_start + win_length <= num_frames:
            win_end = win_start + win_length

            # 현재 윈도우에 대한 평균 및 정규화
            rgb_base = np.mean(self.RGB_mean_array[win_start:win_end], axis=0)
            rgb_base += 1e-6
            rgb_norm = self.RGB_mean_array[win_start:win_end] / rgb_base

            Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
            Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]

            # 필터 적용
            Xf = filtfilt(B, A, Xs, axis=0)
            Yf = filtfilt(B, A, Ys)

            std_Yf = np.std(Yf)
            if std_Yf > 1e-6:
                alpha = np.std(Xf) / std_Yf
            else:
                alpha = 0.0

            S_window = Xf - alpha * Yf
            S_window *= windows.hann(win_length)

            # 결과 반영
            COMPUTED_PPG_SIGNAL[win_start:win_end] += S_window

            # 다음 윈도우로 이동
            win_start += step_size

        return COMPUTED_PPG_SIGNAL