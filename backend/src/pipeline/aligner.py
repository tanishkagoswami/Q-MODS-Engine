import pandas as pd
import numpy as np
import wfdb
import logging
from typing import Union, List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesAligner:
    """
    Handles the synchronization of high-frequency 125Hz PhysioNet waveform data 
    with sparse, low-frequency MIMIC-IV clinical tabular data (labs/vitals).
    """
    
    def __init__(self, waveform_freq_hz: int = 125):
        self.freq_hz = waveform_freq_hz
        self.period_ms = 1000 / self.freq_hz  # e.g., 125Hz = 8ms period

    def load_waveform_as_dataframe(self, record_path: str, start_time: pd.Timestamp) -> pd.DataFrame:
        """
        Reads a .dat/.hea waveform file using wfdb and converts it to a Pandas 
        DataFrame with an exact DatetimeIndex.
        
        Args:
            record_path: Path to the WFDB record (without extension).
            start_time: The real-world timestamp when the waveform recording started 
                        (needed to align with hospital charttimes).
        """
        try:
            logger.info(f"Reading WFDB record from {record_path}")
            # Read all channels (ECG, ABP, etc.)
            record = wfdb.rdrecord(record_path)
            
            # Convert signals to DataFrame
            df_wave = pd.DataFrame(record.p_signal, columns=record.sig_name)
            
            # Generate the high-frequency DatetimeIndex
            # Using timedelta to add 8ms increments to the start_time
            time_deltas = pd.to_timedelta(np.arange(len(df_wave)) * self.period_ms, unit='ms')
            df_wave.index = start_time + time_deltas
            df_wave.index.name = 'timestamp'
            
            return df_wave
            
        except Exception as e:
            logger.error(f"Failed to load waveform record {record_path}: {e}")
            raise

    def align_labs_to_waveform(self, df_wave: pd.DataFrame, df_labs: pd.DataFrame) -> pd.DataFrame:
        """
        Fuses the sparse lab data into the high-frequency waveform timeline.
        Uses 'merge_asof' to find the closest PREVIOUS lab result for every 125Hz tick.
        
        Args:
            df_wave: High-frequency dataframe (Index must be datetime)
            df_labs: Sparse lab dataframe with a 'charttime' datetime column
            
        Returns:
            A unified DataFrame containing both waveform channels and lab features.
        """
        logger.info("Aligning sparse lab records to high-frequency waveform timeline...")
        
        # Ensure dataframes are sorted by time (required for merge_asof)
        df_wave = df_wave.sort_index()
        df_labs = df_labs.sort_values('charttime')
        
        # Reset index of waveform temporarily so we can merge on it
        df_wave_reset = df_wave.reset_index()
        
        # Merge_asof performs a left join. For every timestamp in df_wave, 
        # it looks backward in df_labs to find the most recent lab value.
        # This acts as a natural "forward fill" physically anchored to time.
        aligned_df = pd.merge_asof(
            df_wave_reset, 
            df_labs, 
            left_on='timestamp', 
            right_on='charttime', 
            direction='backward'
        )
        
        # Restore the DatetimeIndex
        aligned_df.set_index('timestamp', inplace=True)
        aligned_df.drop(columns=['charttime'], inplace=True, errors='ignore')
        
        logger.info(f"Alignment complete. Unified shape: {aligned_df.shape}")
        return aligned_df