import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
import os

class ECGProcessor:
    """
    A class to load, clean, and prepare WFDB ECG data for a processing pipeline.
    """
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir

    def load_record(self, record_name):
        """Loads .hea and .dat files using the wfdb library."""
        record_path = os.path.join(self.raw_data_dir, record_name)
        # Load the record and the annotation/signal data
        record = wfdb.rdrecord(record_path)
        return record.p_signal, record.fs

    def bandpass_filter(self, data, lowcut=0.5, highcut=45.0, fs=250, order=4):
        """Applies a Butterworth bandpass filter to clean the ECG signal."""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        return filtfilt(b, a, data, axis=0)

    def process(self, record_name):
        """Full pipeline: Load -> Filter -> Return prepared array."""
        print(f"Processing record: {record_name}")
        
        # 1. Load
        raw_signal, fs = self.load_record(record_name)
        
        # 2. Preprocess (Filter)
        # Note: If you have multiple leads, this processes all of them
        clean_signal = self.bandpass_filter(raw_signal, fs=fs)
        
        print(f"Pipeline complete. Signal shape: {clean_signal.shape}")
        return clean_signal, fs

# --- Example Usage ---
# --- Example Usage ---
if __name__ == "__main__":
    # Point this to your directory
    DATA_DIR = r"C:\Users\tanis\Q-MODS-Engine\backend\data\raw"
    PROCESSED_DIR = r"C:\Users\tanis\Q-MODS-Engine\backend\data\processed"
    
    FILENAME = "patient_1001_layout" 
    
    if os.path.exists(os.path.join(DATA_DIR, f"{FILENAME}.hea")):
        processor = ECGProcessor(DATA_DIR)
        clean_data, sampling_rate = processor.process(FILENAME)
        
        # --- NEW: Save the cleaned data ---
        os.makedirs(PROCESSED_DIR, exist_ok=True) # Make sure the processed folder exists
        save_path = os.path.join(PROCESSED_DIR, f"{FILENAME}_clean.npy")
        np.save(save_path, clean_data)
        
        print(f"Success! Saved clean data to: {save_path}")
    else:
        print(f"File {FILENAME} not found in {DATA_DIR}. Please check the filename.")