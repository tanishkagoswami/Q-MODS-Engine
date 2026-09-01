import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysiologyGNNExplainer:
    """
    Custom GNN Explainer tailored for the spatiotemporal physiology graph.
    Optimizes a continuous edge mask to isolate which directed organ-to-organ 
    pathways are driving a patient's systemic cascade alert.
    """

    # Static mapping of organ nodes for clear explanation outputs
    ORGAN_MAP = {
        0: "Renal",
        1: "Pulmonary",
        2: "Cardiovascular",
        3: "Hepatic",
        4: "CNS"
    }

    def __init__(self, model: nn.Module, epochs: int = 100, lr: float = 0.05):
        """
        Args:
            model (nn.Module): The trained STGCN model.
            epochs (int): Number of optimization steps to find the edge mask.
            lr (float): Learning rate for the mask optimizer.
        """
        self.model = model
        self.model.eval()  # Keep model in evaluation mode
        self.epochs = epochs
        self.lr = lr

    def explain_cascade(self, history_states: torch.Tensor, edge_index: torch.Tensor, 
                        edge_weight: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Optimizes a continuous mask over the edges to identify the primary failure pathways.
        
        Args:
            history_states (Tensor): Input sequence [Batch_Size=1, Num_Nodes=5, Seq_Len, In_Features]
            edge_index (Tensor): Base graph structure [2, Num_Edges=9]
            edge_weight (Tensor): Dynamic attention weights [Num_Edges, 1] or [Num_Edges]
            
        Returns:
            List[Dict]: List of active pathways sorted by their calculated importance.
        """
        logger.info("Initializing GNN edge-mask optimization...")
        num_edges = edge_index.size(1)

        # Force input to have a batch dimension of 1 if not present
        if len(history_states.shape) == 3:
            history_states = history_states.unsqueeze(0)

        # Get original unmasked prediction from the model
        with torch.no_grad():
            _, original_cascade_score = self.model(history_states, edge_index, edge_weight.squeeze())
            original_cascade_score = original_cascade_score.item()

        # Initialize raw learnable mask parameters
        # We start with small random values so the sigmoid puts the starting mask close to 0.5
        raw_mask = torch.randn(num_edges, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([raw_mask], lr=self.lr)

        # Optimization loop to find the minimal subgraph mask that explains the prediction
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Bound the mask elements strictly between 0.0 and 1.0 using sigmoid
            mask = torch.sigmoid(raw_mask)
            
            # Apply mask to original dynamic edge weights
            # edge_weight has shape [Num_Edges, 1] or [Num_Edges]
            masked_weights = edge_weight.squeeze() * mask
            
            # Pass through the model with masked weights
            _, masked_cascade_score = self.model(history_states, edge_index, masked_weights)
            
            # ---- Loss Function Objectives ----
            # 1. Prediction Fidelity Loss: The masked graph output should match the original prediction
            pred_loss = torch.pow(masked_cascade_score - original_cascade_score, 2)
            
            # 2. Sparsity Loss (L1 Regularization): Minimize the sum of the mask values to find the smallest bottleneck
            sparsity_loss = torch.sum(mask)
            
            # 3. Entropy Regularization: Encourage mask values to push toward solid 0.0 or 1.0 bounds
            entropy_loss = -torch.sum(mask * torch.log(mask + 1e-6) + (1.0 - mask) * torch.log(1.0 - mask + 1e-6))
            
            # Combine losses with strict regularizing hyperparameter weights
            total_loss = pred_loss + 0.05 * sparsity_loss + 0.02 * entropy_loss
            
            total_loss.backward()
            optimizer.step()

        # Extract the final optimized edge mask
        with torch.no_grad():
            final_mask = torch.sigmoid(raw_mask).cpu().numpy()

        # Map edge weights to clinical explanations
        pathway_attributions = []
        for i in range(num_edges):
            src_node = int(edge_index[0, i].item())
            tgt_node = int(edge_index[1, i].item())
            
            src_name = self.ORGAN_MAP.get(src_node, f"Organ_{src_node}")
            tgt_name = self.ORGAN_MAP.get(tgt_node, f"Organ_{tgt_node}")
            
            pathway_attributions.append({
                "source_organ": src_name,
                "target_organ": tgt_name,
                "pathway_name": f"{src_name} → {tgt_name}",
                "importance_weight": float(final_mask[i]),
                "dynamic_flow_rate": float(edge_weight[i].item())
            })

        # Sort pathways by optimized mask weight (highest impact first)
        pathway_attributions.sort(key=lambda x: x["importance_weight"], reverse=True)
        
        logger.info("GNN Explainer optimization complete.")
        return pathway_attributions