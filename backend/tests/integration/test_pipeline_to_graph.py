import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pipeline.aligner import TimeSeriesAligner
from src.pipeline.denoiser import SignalDenoiser
from src.pipeline.imputer import ICUDataImputer
from src.graph.physiology_graph import PhysiologyGraphBuilder

class TestPipelineToGraphIntegration(unittest.TestCase):
    """Integration test simulating the entire workflow from raw clinical data to graph structures."""

    def test_integration_flow(self):
        start_time = pd.Timestamp("2026-07-02 12:00:00")
        
        # Step 1: Create raw noisy high-frequency waveforms (125Hz)
        time_deltas = pd.to_timedelta(np.arange(125) * (1000/125), unit='ms')
        df_wave = pd.DataFrame({
            "ECG": np.sin(np.linspace(0, 10, 125)) + 0.1 * np.sin(np.linspace(0, 500, 125)), # with 60Hz hum
            "ABP": np.linspace(120, 115, 125)
        }, index=start_time + time_deltas)
        
        # Step 2: Create raw sparse clinical labs
        df_labs = pd.DataFrame({
            "charttime": [start_time, start_time + pd.Timedelta(seconds=5)],
            "creatinine": [1.2, np.nan],  # includes missing entry
            "spo2": [0.97, 0.98]
        })

        # --- Execution of Pipeline ---
        
        # A. Denoise Waveform
        denoiser = SignalDenoiser(fs=125.0)
        df_wave_clean = denoiser.denoise_waveform_dataframe(df_wave)
        self.assertEqual(df_wave_clean.shape, df_wave.shape)

        # B. Align Time Series
        aligner = TimeSeriesAligner(waveform_freq_hz=125)
        df_aligned = aligner.align_labs_to_waveform(df_wave_clean, df_labs)
        
        # C. Impute Missing Values
        imputer = ICUDataImputer()
        df_imputed = imputer.process_patient_pipeline(
            df_aligned, 
            lab_cols=["creatinine", "spo2"], 
            waveform_cols=["ECG", "ABP"]
        )
        
        # Verify MICE and clinical forward fill completed successfully
        self.assertFalse(df_imputed["creatinine"].isnull().any())

        # D. Convert to Spatial Physiology Graph
        graph_builder = PhysiologyGraphBuilder()
        patient_slice = df_imputed.iloc[-1].to_dict()
        graph_data = graph_builder.build_graph(patient_slice, sequence_id=42)

        # Assertions on final graph format
        self.assertEqual(graph_data.x.shape, (5, 3))  # 5 nodes with 3 clinical features each
        self.assertEqual(graph_data.edge_index.shape[0], 2)  # [2, num_edges]

if __name__ == "__main__":
    unittest.main()