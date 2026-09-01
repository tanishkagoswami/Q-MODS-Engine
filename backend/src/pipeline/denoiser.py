import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import logging
from typing import Union, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalDenoiser:
    """
    Applies digital filters to high-frequency ICU waveforms (ECG, ABP, Respiration)
    to remove powerline interference, baseline wander, and high-frequency noise.
    """
    
    def __init__(self, fs: float = 125.0):
        """
        Args:
            fs (float): Sampling frequency of the signals in Hz. MIMIC-IV defaults to 125Hz.
        """
        self.fs = fs

    def apply_notch_filter(self, signal: np.ndarray, cutoff_freq: float = 60.0, quality_factor: float = 30.0) -> np.ndarray:
        """
        Removes AC powerline hum (typically 60Hz in the US/MIMIC data) using an IIR Notch filter.
        """
        if np.all(np.isnan(signal)):
            return signal
            
        # Nyquist frequency
        nyq = 0.5 * self.fs
        w0 = cutoff_freq / nyq
        
        # Design notch filter
        b, a = iirnotch(w0, quality_factor)
        
        # zero-phase filtering (filtfilt forward and backward so there's no phase shift)
        # Using nan_to_num temporarily if minor gaps exist, to avoid filter explosion
        clean_signal = filtfilt(b, a, np.nan_to_num(signal))
        return clean_signal

    def apply_butterworth_filter(self, signal: np.ndarray, btype: str = 'bandpass', 
                                 lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
        """
        Applies a Butterworth filter. 
        - 'bandpass' is ideal for ECG (0.5 to 40Hz) to remove both muscle noise and breathing artifacts.
        - 'lowpass' is ideal for Arterial Blood Pressure (ABP) to smooth high-frequency ripples.
        """
        if np.all(np.isnan(signal)):
            return signal

        nyq = 0.5 * self.fs
        
        if btype == 'bandpass':
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype=btype)
        elif btype == 'lowpass':
            high = highcut / nyq
            b, a = butter(order, high, btype=btype)
        elif btype == 'highpass':
            low = lowcut / nyq
            b, a = butter(order, low, btype=btype)
        else:
            raise ValueError(f"Invalid filter type: {btype}. Choose 'lowpass', 'highpass', or 'bandpass'.")

        # Apply zero-phase filter
        return filtfilt(b, a, np.nan_to_num(signal))

    def denoise_waveform_dataframe(self, df: pd.DataFrame, ecg_channels: List[str] = ['ECG', 'II'], 
                                   abp_channels: List[str] = ['ABP', 'ART']) -> pd.DataFrame:
        """
        Takes a multi-channel waveform DataFrame and applies targeted filtering 
        based on the clinical channel type.
        """
        df_clean = df.copy()
        
        for col in df.columns:
            # Match channel types ignoring case sensitivity
            if any(x.lower() in col.lower() for x in ecg_channels):
                logger.info(f"Applying Bandpass (0.5-40Hz) + Notch (60Hz) filter to ECG channel: {col}")
                # 1. Remove 60Hz hum
                signal_notch = self.apply_notch_filter(df_clean[col].values, cutoff_freq=60.0)
                # 2. Bandpass filter to isolate core ECG features
                df_clean[col] = self.apply_butterworth_filter(signal_notch, btype='bandpass', lowcut=0.5, highcut=40.0)
                
            elif any(x.lower() in col.lower() for x in abp_channels):
                logger.info(f"Applying Lowpass (15Hz) filter to Blood Pressure channel: {col}")
                # ABP doesn't need highpass filtering, just lowpass to smooth artifacts
                df_clean[col] = self.apply_butterworth_filter(df_clean[col].values, btype='lowpass', highcut=15.0)
                
        return df_clean