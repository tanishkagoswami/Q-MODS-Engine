import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalGatedConv(nn.Module):
    """
    Temporal Gated Convolutional Layer (1D Casual Conv + Gated Linear Unit).
    Captures temporal progression patterns within each individual organ system node.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalGatedConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Causal padding
        
        # Convolutions for the gate mechanism (GLU)
        # We output twice the channels because half will act as the gate filter
        self.conv = nn.Conv1d(
            in_channels, out_channels * 2, kernel_size=kernel_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape [Batch_Size, Features, Seq_Length]
        Returns:
            Tensor: Shape [Batch_Size, Out_Channels, Seq_Length] after GLU
        """
        # Apply manual right-side causal padding so future values do not leak
        x_padded = F.pad(x, (self.padding, 0))
        
        # Convolve
        conv_out = self.conv(x_padded)
        
        # Split channels for GLU (Gated Linear Unit) activation
        # GLU: P * sigmoid(Q)
        lhs, rhs = torch.chunk(conv_out, 2, dim=1)
        return lhs * torch.sigmoid(rhs)

class DenseGCNLayer(nn.Module):
    """Same maths as torch_geometric's GCNConv (self-loops + symmetric normalisation)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ch))
    def forward(self, x, edge_index, edge_weight):
        # x: [N, C], edge_index: [2, E] (row 0 = source, row 1 = target), edge_weight: [E]
        n = x.size(0)
        A = torch.zeros(n, n, dtype=x.dtype, device=x.device)
        A[edge_index[1], edge_index[0]] = edge_weight.to(x.dtype)  # message goes source -> target
        A = A + torch.eye(n, dtype=x.dtype, device=x.device)        # self-loops with weight 1
        d = A.sum(dim=1).pow(-0.5)
        d[torch.isinf(d)] = 0
        return (d[:, None] * A * d[None, :]) @ self.lin(x) + self.bias
class STGCNBlock(nn.Module):
    """
    A single Spatiotemporal Graph Convolutional Block.
    Executes a sandwich structure: Temporal Conv -> Spatial Graph Conv -> Temporal Conv.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_nodes: int = 5):
        super(STGCNBlock, self).__init__()
        self.num_nodes = num_nodes
        
        # First temporal gated convolution
        self.temp_conv1 = TemporalGatedConv(in_channels, hidden_channels)
        
        # Spatial Graph Convolution using PyTorch Geometric's GCNConv
        # Propagates failure states between connected organ nodes
        self.spatial_gcn = DenseGCNLayer(hidden_channels, hidden_channels)        
        # Second temporal gated convolution
        self.temp_conv2 = TemporalGatedConv(hidden_channels, out_channels)
        
        # Normalization layers
        self.layer_norm = nn.LayerNorm([out_channels, num_nodes])
        
        # Skip connection if input/output channels differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Node states of shape [Batch_Size, Num_Nodes, Seq_Length, In_Channels]
            edge_index (Tensor): Graph edge indices [2, Num_Edges]
            edge_weight (Tensor): Dynamic attention weights [Num_Edges, 1] or [Num_Edges]
            
        Returns:
            Tensor: Updated node states [Batch_Size, Num_Nodes, Seq_Length, Out_Channels]
        """
        batch_size, num_nodes, seq_len, in_c = x.shape
        
        # Save residual
        res = x.permute(0, 3, 1, 2).reshape(batch_size * num_nodes, in_c, seq_len)
        if self.residual is not None:
            res = self.residual(res)
        res = res.reshape(batch_size, -1, num_nodes, seq_len).permute(0, 2, 3, 1) # [B, N, T, Out_C]

        # ---- Step 1: First Temporal Layer ----
        # Reshape to [Batch_Size * Num_Nodes, In_Channels, Seq_Length]
        t1 = x.permute(0, 1, 3, 2).reshape(batch_size * num_nodes, in_c, seq_len)
        t1_out = self.temp_conv1(t1) # [B * N, Hidden_C, T]
        
        # ---- Step 2: Spatial Graph Layer ----
        # Reshape to [Batch_Size, Num_Nodes, Seq_Length, Hidden_Channels]
        s1 = t1_out.reshape(batch_size, num_nodes, -1, seq_len).permute(0, 3, 1, 2) # [B, T, N, Hidden_C]
        s_out_list = []
        
        # We propagate spatially across graph nodes for each temporal slice
        for t in range(seq_len):
            s_t = s1[:, t, :, :] # [Batch_Size, Num_Nodes, Hidden_Channels]
            s_t_fused = []
            for b in range(batch_size):
                # Run PyG Spatial Convolution using dynamic attention-based edge weights
                out_gcn = self.spatial_gcn(s_t[b], edge_index, edge_weight)
                s_t_fused.append(out_gcn)
            s_out_list.append(torch.stack(s_t_fused)) # Stack batches
            
        # Reshape spatial outputs back to temporal format: [Batch_Size * Num_Nodes, Hidden_Channels, Seq_Length]
        s_out = torch.stack(s_out_list, dim=2) # [B, N, T, Hidden_C]
        s_out_trans = s_out.permute(0, 1, 3, 2).reshape(batch_size * num_nodes, -1, seq_len)

        # ---- Step 3: Second Temporal Layer ----
        t2_out = self.temp_conv2(s_out_trans) # [B * N, Out_Channels, T]
        
        # Reshape to [Batch_Size, Out_Channels, Num_Nodes, Seq_Length]
        final_out = t2_out.reshape(batch_size, num_nodes, -1, seq_len).permute(0, 2, 1, 3) # [B, Out_C, N, T]
        
        # Apply layer normalization and residual addition
        normalized = self.layer_norm(final_out.permute(0, 3, 1, 2).reshape(-1, final_out.size(1), num_nodes))
        normalized = normalized.reshape(batch_size, seq_len, -1, num_nodes).permute(0, 3, 1, 2) # [B, N, T, Out_C]
        
        return F.relu(normalized + res)


class STGCN(nn.Module):
    """
    Complete Spatiotemporal Graph Convolutional Network (STGCN).
    Processes history to predict probability/severity of organ failure across 5 systems.
    """
    def __init__(self, in_features: int = 128, hidden_dim: int = 64, out_features: int = 32, num_nodes: int = 5):
        """
        Args:
            in_features (int): Dimension of input patient state vector (default: 128).
            hidden_dim (int): Intermediate processing dimensions.
            out_features (int): Representation size before final projection.
            num_nodes (int): Number of organs in the graph (default: 5).
        """
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        
        # Pre-mapping layer to align input features per node
        self.node_pre_map = nn.Linear(in_features, 16)
        
        # Stacked STGCN Block
        self.stgcn_block = STGCNBlock(
            in_channels=16, 
            hidden_channels=hidden_dim, 
            out_channels=out_features, 
            num_nodes=num_nodes
        )
        
        # Output multi-task projection heads
        # 1. Predicts severity trajectory [0.0 - 1.0] for each of the 5 organs
        self.failure_regression_head = nn.Linear(out_features, 1)
        # 2. Predicts global likelihood of a full systemic crash cascade within 4 hours
        self.cascade_alert_head = nn.Linear(num_nodes * out_features, 1)

    def forward(self, history_states: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history_states (Tensor): Historical sequence of patient state vectors 
                                     Shape: [Batch_Size, Num_Nodes, Seq_Length, In_Features]
                                     e.g., [32, 5, 24, 128] (last 24 hours of 128-dim vectors)
            edge_index (Tensor): Dynamic graph edges [2, Num_Edges]
            edge_weight (Tensor): Dynamic attention weights [Num_Edges]
            
        Returns:
            Tuple[Tensor, Tensor]: 
                - Organ-specific failure trajectories: [Batch_Size, 5] (values bounded [0, 1])
                - Global systemic cascade risk score: [Batch_Size, 1] (value bounded [0, 1])
        """
        batch_size, num_nodes, seq_len, in_f = history_states.shape
        
        # Step 1: Down-project high-dimensional inputs to fit network size
        # Shape: [Batch_Size, Num_Nodes, Seq_Length, 16]
        projected_states = self.node_pre_map(history_states)
        
        # Step 2: Feed sequence through Spatiotemporal blocks
        # Shape: [Batch_Size, Num_Nodes, Seq_Length, Out_Features]
        features = self.stgcn_block(projected_states, edge_index, edge_weight)
        
        # Step 3: Global Temporal Pooling (we take the state at the latest timestamp)
        # Shape: [Batch_Size, Num_Nodes, Out_Features]
        latest_features = features[:, :, -1, :]
        
        # Step 4: Multi-Task Classification/Regression Output
        # Organ failure trajectories (individual hazard scores for each of the 5 nodes)
        # Shape: [Batch_Size, 5]
        organ_hazards = torch.sigmoid(self.failure_regression_head(latest_features)).squeeze(-1)
        
        # Flatten organ representations to evaluate cross-organ cascade metrics globally
        # Shape: [Batch_Size, 5 * Out_Features]
        flattened_reps = latest_features.reshape(batch_size, -1)
        cascade_alert_score = torch.sigmoid(self.cascade_alert_head(flattened_reps))
        
        return organ_hazards, cascade_alert_score