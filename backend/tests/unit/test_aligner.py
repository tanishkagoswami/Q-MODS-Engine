import unittest
import pandas as pd
import numpy as np
import sys
import os

# Align workspace pathing so tests can import from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pipeline.aligner import TimeSeriesAligner

class TestTimeSeriesAligner(unittest.TestCase):
    """Unit tests for aligning high-frequency waveforms with sparse clinical labs."""

    def setUp(self):
        self.aligner = TimeSeriesAligner(waveform_freq_hz=125)
        self.start_time = pd.Timestamp("2026-07-02 12:00:00")

    def test_align_labs_to_waveform(self):
        # Create a mock 125Hz waveform dataframe (3 samples = 16ms span)
        time_deltas = pd.to_timedelta([0, 8, 16], unit='ms')
        df_wave = pd.DataFrame(
            {"ECG": [0.5, 0.6, 0.55], "ABP": [120, 118, 119]},
            index=self.start_time + time_deltas
        )

        # Create sparse lab data. First lab drawn at start_time, second 1 hour later
        df_labs = pd.DataFrame({
            "charttime": [self.start_time, self.start_time + pd.Timedelta(hours=1)],
            "creatinine": [1.1, 1.4],
            "spo2": [0.98, 0.95]
        })

        # Align
        df_aligned = self.aligner.align_labs_to_waveform(df_wave, df_labs)

        # Assertions
        self.assertEqual(len(df_aligned), 3)
        self.assertIn("creatinine", df_aligned.columns)
        self.assertIn("spo2", df_aligned.columns)
        
        # Verify backward-merge (last observation carried forward) behavior
        # Since all 3 samples occur within milliseconds of self.start_time,
        # they should all inherit the first lab values (creatinine=1.1)
        self.assertAlmostEqual(df_aligned["creatinine"].iloc[0], 1.1)
        self.assertAlmostEqual(df_aligned["creatinine"].iloc[2], 1.1)

if __name__ == "__main__":
    unittest.main()