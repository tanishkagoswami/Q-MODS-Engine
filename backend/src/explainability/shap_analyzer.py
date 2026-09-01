import torch
import numpy as np
import pandas as pd
import shap
import logging
from typing import List, Dict, Union, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ICUBiomarkerSHAPAnalyzer:
    """
    Computes SHAP (SHapley Additive exPlanations) attribution values for clinical biomarkers.
    Identifies which lab results or vitals contributed most to the organ failure and cascade alerts.
    """

    def __init__(self, model_predict_fn: Callable[[np.ndarray], np.ndarray], 
                 feature_names: List[str], background_data: np.ndarray):
        """
        Args:
            model_predict_fn: A callable function that takes a 2D numpy array of features 
                              and returns model predictions (e.g., cascade risk scores).
            feature_names: List of human-readable biomarker names matching the input features.
            background_data: A representative baseline sample of patient states [N, Num_Features]
                             used to build the SHAP background distribution.
        """
        self.model_predict_fn = model_predict_fn
        self.feature_names = feature_names
        self.background_data = background_data
        
        # We use KernelExplainer as it is model-agnostic, which is crucial for 
        # explaining hybrid classical-quantum models that standard PyTorch deep explainers struggle with.
        # We summarize the background data using shap.kmeans to keep execution fast and real-time.
        logger.info("Initializing SHAP Kernel Explainer with background distribution...")
        summarized_background = shap.kmeans(background_data, k=10) if len(background_data) > 10 else background_data
        self.explainer = shap.KernelExplainer(self.model_predict_fn, summarized_background)

    def compute_attributions(self, patient_features: np.ndarray) -> Dict[str, Union[List[float], float]]:
        """
        Calculates local SHAP values for a single patient slice.
        
        Args:
            patient_features (np.ndarray): Shape [1, Num_Features] or [Num_Features]
            
        Returns:
            Dict: Mappings of feature names to their absolute SHAP attribution values
                  and the base expected value of the network.
        """
        # Ensure correct input dimensions
        if len(patient_features.shape) == 1:
            patient_features = patient_features.reshape(1, -1)
            
        logger.info("Computing SHAP values for patient slice...")
        # Compute SHAP values for the sample
        shap_values = self.explainer.shap_values(patient_features, silent=True)
        
        # Handle multi-task or single-task outputs
        # shap_values can be a list of arrays (one per output class/head) or a single array
        if isinstance(shap_values, list):
            # If multi-output, default to explaining the first prediction head (e.g., cascade risk)
            local_shap = shap_values[0][0]
        else:
            local_shap = shap_values[0]
            
        # Map feature names to their respective attribution scores
        attributions = {self.feature_names[i]: float(local_shap[i]) for i in range(len(self.feature_names))}
        
        # Extract the explainer base value (the baseline risk before seeing features)
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])
            
        return {
            "attributions": attributions,
            "base_value": float(base_val)
        }

    def generate_clinical_explanation(self, patient_features: np.ndarray) -> Dict[str, any]:
        """
        Generates a structured medical explanation detailing the top exacerbating and stabilizing factors.
        """
        res = self.compute_attributions(patient_features)
        attribs = res["attributions"]
        
        # Sort features based on direction and magnitude
        exacerbating = []
        stabilizing = []
        
        for name, val in attribs.items():
            feature_data = {
                "biomarker": name,
                "impact_magnitude": abs(val),
                "direction": "worsening" if val > 0 else "improving"
            }
            if val > 0:
                exacerbating.append(feature_data)
            else:
                stabilizing.append(feature_data)
                
        # Sort collections by absolute impact
        exacerbating.sort(key=lambda x: x["impact_magnitude"], reverse=True)
        stabilizing.sort(key=lambda x: x["impact_magnitude"], reverse=True)
        
        return {
            "base_alert_probability": res["base_value"],
            "top_driving_factors_of_failure": exacerbating[:3],  # Top 3 worsening markers
            "top_protective_factors": stabilizing[:3]            # Top 3 stabilizing markers
        }