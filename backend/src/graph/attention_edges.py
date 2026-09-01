import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysiologicalAttentionEdges(nn.Module):
    """
    Computes temporal attention weights across organ system nodes over time.
    Translates raw multi-dimensional physiological states into dynamic edge attributes
    representing the directional velocity of cross-organ failure cascades.
    """
    
    def __init__(self, node_feat_dim: int = 3, embed_dim: int = 16, num_heads: int = 2):
        """
        Args:
            node_feat_dim (int): Dimensionality of raw organ features (default: 3).
            embed_dim (int): Projection dimension for query/key/value calculations.
            num_heads (int): Number of attention heads for clinical multi-faceted patterns.
        """
        super(PhysiologicalAttentionEdges, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Linear projection layers for Query, Key, and Value
        self.q_proj = nn.Linear(node_feat_dim, embed_dim)
        self.k_proj = nn.Linear(node_feat_dim, embed_dim)
        self.v_proj = nn.Linear(node_feat_dim, embed_dim)
        
        # Scaling factor for scaled dot-product attention
        self.scale = 1.0 / (embed_dim ** 0.5)
        
        # Output layer to project fused attention values back to edge coefficients
        self.edge_out = nn.Linear(embed_dim, 1)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Computes dynamic edge weights based on physical organ node correlations.
        
        Args:
            node_features (torch.Tensor): Shape [Batch_Size, Num_Nodes, Node_Feat_Dim]
                                          or [Num_Nodes, Node_Feat_Dim]
            edge_index (torch.Tensor): Connection structure of the graph [2, Num_Edges]
            
        Returns:
            torch.Tensor: Normalized edge weights of shape [Num_Edges, 1], representing
                          active hazard multipliers.
        """
        # Ensure input tensor has batch dimension [B, N, F]
        if len(node_features.shape) == 2:
            node_features = node_features.unsqueeze(0)  # Shape: [1, 5, 3]
            
        batch_size, num_nodes, feat_dim = node_features.shape
        num_edges = edge_index.size(1)
        
        # Project node features to Query, Key, and Value spaces
        # Shape: [Batch_Size, Num_Nodes, Embed_Dim]
        queries = self.q_proj(node_features)
        keys = self.k_proj(node_features)
        
        # Initialize an empty tensor for edge weights
        # We process batches efficiently but construct output mapped explicitly to the graph index
        dynamic_edge_attrs = []
        
        for b in range(batch_size):
            b_queries = queries[b]  # [Num_Nodes, Embed_Dim]
            b_keys = keys[b]        # [Num_Nodes, Embed_Dim]
            
            # Map edge_index elements to extract source and target states
            # edge_index[0] contains source node indices, edge_index[1] contains target indices
            sources = edge_index[0]  # [Num_Edges]
            targets = edge_index[1]  # [Num_Edges]
            
            # Extract source Query vectors and target Key vectors
            q_src = b_queries[sources]  # [Num_Edges, Embed_Dim]
            k_tgt = b_keys[targets]     # [Num_Edges, Embed_Dim]
            
            # Scaled Dot-Product calculation for directed edges
            # Measures how much the source node is destabilizing or "dragging down" the target node
            scores = torch.sum(q_src * k_tgt, dim=-1) * self.scale  # [Num_Edges]
            
            # We apply a soft sigmoid to scale our dynamic weights gracefully between 0.1 and 3.0
            # rather than Softmax, because multiple independent organ systems can crash simultaneously
            # (non-exclusive attention).
            edge_weights = 0.1 + 2.9 * torch.sigmoid(scores)  # [Num_Edges, 1]
            dynamic_edge_attrs.append(edge_weights)
            
        # Stack output over batch dimension
        final_edge_attr = torch.stack(dynamic_edge_attrs, dim=0)  # [Batch_Size, Num_Edges, 1]
        
        # Squeeze batch dimension if input was single-instance
        if batch_size == 1:
            final_edge_attr = final_edge_attr.squeeze(0)          # [Num_Edges, 1]
            
        return final_edge_attr