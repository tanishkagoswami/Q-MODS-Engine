import torch
import types
import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysiologyGraphBuilder:
    """
    Builds a PyTorch Geometric (PyG) Dynamic Graph mapping 5 primary organ systems.
    Defines directed edges based on known pathophysiological pathways of cross-organ failure.
    """
    
    # Define our static node mappings for consistent tracking
    ORGAN_MAP = {
        0: "Renal",
        1: "Pulmonary",
        2: "Cardiovascular",
        3: "Hepatic",
        4: "CNS"
    }
    
    def __init__(self):
        # Establish the base clinical connection map (directed physiological pathways)
        # Pairs are defined as (source_organ_idx, target_organ_idx)
        # e.g., (2, 0) represents Cardiovascular -> Renal (Cardiorenal Syndrome)
        self.base_edges = torch.tensor([
            [2, 0],  # Cardiovascular -> Renal (Hypoperfusion leading to acute kidney injury)
            [0, 2],  # Renal -> Cardiovascular (Uremic cardiomyopathy, fluid overload)
            [1, 2],  # Pulmonary -> Cardiovascular (Hypoxemia causing myocardial ischemia)
            [2, 1],  # Cardiovascular -> Pulmonary (Congestive heart failure causing pulmonary edema)
            [3, 0],  # Hepatic -> Renal (Hepatorenal syndrome via splanchnic vasodilation)
            [2, 4],  # Cardiovascular -> CNS (Low cardiac output causing encephalopathy)
            [0, 4],  # Renal -> CNS (Uremic encephalopathy)
            [3, 4],  # Hepatic -> CNS (Hepatic encephalopathy via ammonia buildup)
            [1, 4]   # Pulmonary -> CNS (Hypercapnia/Hypoxia causing cerebral dysfunction)
        ], dtype=torch.long).t().contiguous()  # Reshape to [2, Num_Edges] for PyG

    def map_features_to_nodes(self, patient_slice: Dict[str, float]) -> torch.Tensor:
        """
        Groups clinical features into a dedicated feature matrix (X) of shape [5, Num_Features_Per_Node].
        Each organ system node receives its respective real-time biomarkers.
        
        If a feature is missing or not supplied, we default to 0.0 (assuming it has been imputed).
        """
        # Node 0: Renal (Kidneys) -> [Creatinine, BUN, Urine Output]
        renal_feats = [
            patient_slice.get("creatinine", 0.0),
            patient_slice.get("bun", 0.0),
            patient_slice.get("urine_output", 0.0)
        ]
        
        # Node 1: Pulmonary (Lungs) -> [SpO2, Respiratory Rate, PaO2/FiO2 ratio]
        pulm_feats = [
            patient_slice.get("spo2", 0.0),
            patient_slice.get("respiratory_rate", 0.0),
            patient_slice.get("pao2_fio2", 0.0)
        ]
        
        # Node 2: Cardiovascular (Heart) -> [Heart Rate, MAP, Systolic BP]
        cardio_feats = [
            patient_slice.get("heart_rate", 0.0),
            patient_slice.get("mean_arterial_pressure", 0.0),
            patient_slice.get("systolic_bp", 0.0)
        ]
        
        # Node 3: Hepatic (Liver) -> [Total Bilirubin, ALT, AST]
        hep_feats = [
            patient_slice.get("bilirubin_total", 0.0),
            patient_slice.get("alt", 0.0),
            patient_slice.get("ast", 0.0)
        ]
        
        # Node 4: CNS (Brain) -> [Glasgow Coma Scale (GCS), Pupillary response, Temperature]
        cns_feats = [
            patient_slice.get("gcs", 15.0),  # Default healthy GCS is 15
            patient_slice.get("pupil_reaction", 1.0),
            patient_slice.get("temperature", 37.0)
        ]
        
        # Assemble into a PyTorch FloatTensor of shape [5, 3]
        x = torch.tensor([
            renal_feats,
            pulm_feats,
            cardio_feats,
            hep_feats,
            cns_feats
        ], dtype=torch.float)
        
        return x

    def build_graph(self, patient_slice: Dict[str, float], sequence_id: int = 0) -> types.SimpleNamespace:
            """
            Constructs a lightweight SimpleNamespace object containing:
            - x: Node feature matrix [5, 3]
            - edge_index: Graph connection indices [2, 9]
            - edge_attr: Dynamic edge weights initialized to baseline (1.0)
            """
            # Step 1: Map clinical features to our 5 organ nodes
            x = self.map_features_to_nodes(patient_slice)
            
            # Step 2: Initialize edge attributes (weights)
            # We start with baseline edge attributes of 1.0. These will be scaled later
            # in Layer 2's temporal attention modules to show failure velocity.
            num_edges = self.base_edges.size(1)
            edge_attr = torch.ones((num_edges, 1), dtype=torch.float)
            
            # Step 3: Construct the lightweight structure instead of PyG Data
            graph_data = types.SimpleNamespace(
                x=x,
                edge_index=self.base_edges,
                edge_attr=edge_attr,
                seq_id=torch.tensor([sequence_id], dtype=torch.long)
            )
            
            return graph_data

    def get_organ_name(self, node_idx: int) -> str:
        """Helper to return human-readable organ names for explainability"""
        return self.ORGAN_MAP.get(node_idx, "Unknown")