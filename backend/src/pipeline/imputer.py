import numpy as np
import pandas as pd
import logging
from typing import List, Optional

# IterativeImputer is still experimental in scikit-learn, so we must import enable_iterative_imputer first
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ICUDataImputer:
    """
    Imputation pipeline designed specifically for hybrid high-frequency and sparse ICU data.
    Combines clinical forward-filling (LOCF) with Multivariate Imputation by Chained 
    Equations (MICE) to fill starting gaps and missing structural labs.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 42):
        """
        Args:
            max_iter (int): Maximum number of imputation rounds for MICE.
            random_state (int): Seed for reproducibility in Bayesian Ridge regression.
        """
        self.max_iter = max_iter
        self.random_state = random_state
        # We use BayesianRidge as our estimator for MICE to handle multicollinearity among lab features
        self.mice_estimator = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order='ascending'
        )

    def impute_sensor_dropouts(self, df: pd.DataFrame, limit_ms: int = 1000, fs: float = 125.0) -> pd.DataFrame:
        """
        For high-frequency waveforms (ECG/ABP), minor gaps are usually temporary physical dropouts 
        (e.g., patient moving). We use cubic/linear spline interpolation for brief drops.
        
        Args:
            df: Waveform DataFrame
            limit_ms: Maximum duration of a dropout to interpolate (default 1000ms).
            fs: Sampling rate.
        """
        df_clean = df.copy()
        # Convert time limit to number of indices (e.g., 1000ms at 125Hz = 125 steps)
        limit_steps = int((limit_ms / 1000.0) * fs)
        
        # Linearly interpolate small gaps. Larger gaps are left as NaN so the model knows
        # there is a physical lead disconnection.
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].interpolate(
                method='linear', 
                limit=limit_steps, 
                limit_direction='both'
            )
        return df_clean

    def apply_clinical_forward_fill(self, df: pd.DataFrame, lab_cols: List[str]) -> pd.DataFrame:
        """
        In clinical environments, a patient's lab value (e.g., potassium) is clinically assumed 
        to stay the same until a new lab is drawn. This implements LOCF (Last Observation Carried Forward).
        """
        df_ffilled = df.copy()
        # Forward fill the lab columns down the high-frequency rows
        df_ffilled[lab_cols] = df_ffilled[lab_cols].ffill()
        return df_ffilled

    def fit_transform_mice(self, df: pd.DataFrame, lab_cols: List[str]) -> pd.DataFrame:
        """
        MICE is used for any remaining NaNs. This is critical for:
        1. 'Cold Start' gaps (the period between ICU admission and the very first lab draw).
        2. Unrecorded labs where relationships with other active labs can predict the value.
        """
        logger.info(f"Running MICE imputation on columns: {lab_cols}")
        df_imputed = df.copy()
        
        # Extract the lab data subset for MICE
        lab_subset = df_imputed[lab_cols].values
        
        # Fit and transform the imputer on the subset
        imputed_subset = self.mice_estimator.fit_transform(lab_subset)
        
        # Assign back to the dataframe
        df_imputed[lab_cols] = imputed_subset
        return df_imputed

    def process_patient_pipeline(self, df: pd.DataFrame, lab_cols: List[str], waveform_cols: List[str]) -> pd.DataFrame:
        """
        Full unified preprocessing sequence for a combined Patient State DataFrame.
        """
        logger.info("Starting complete clinical imputation pipeline...")
        
        # 1. Clean minor physical dropouts on high-frequency channels
        df_processed = df.copy()
        if len(waveform_cols) > 0:
            df_processed[waveform_cols] = self.impute_sensor_dropouts(df_processed[waveform_cols])
            
        # 2. Apply clinical forward-fill for sparse labs along timestamps
        df_processed = self.apply_clinical_forward_fill(df_processed, lab_cols)
        
        # 3. Use MICE to impute any leftover gaps (e.g., before first draw or system errors)
        df_processed = self.fit_transform_mice(df_processed, lab_cols)
        
        # Final safety check: if any minor NaNs remain, fill with median
        if df_processed.isnull().any().any():
            logger.warning("Leftover NaNs detected after MICE. Applying fallback median fill.")
            df_processed = df_processed.fillna(df_processed.median())
            
        logger.info("Imputation pipeline execution finished successfully.")
        return df_processed