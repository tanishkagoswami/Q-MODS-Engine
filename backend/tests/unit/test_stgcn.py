import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.stgcn import STGCN, TemporalGatedConv, STGCNBlock

class TestSTGCN(unittest.TestCase):
    """Unit tests verifying spatiotemporal layers, shapes, and forward steps."""

    def setUp(self):
        self.batch_size = 2
        self.num_nodes = 5
        self.seq_len = 12
        self.in_features = 128
        
        self.model = STGCN(
            in_features=self.in_features,
            hidden_dim=16,
            out_features=8,
            num_nodes=self.num_nodes
        )
        
        # Define mock graph structure matching our 9 physiological pathways
        self.edge_index = torch.tensor([
            [2, 0], [0, 2], [1, 2], [2, 1], [3, 0], [2, 4], [0, 4], [3, 4], [1, 4]
        ], dtype=torch.long).t()
        
        self.edge_weight = torch.ones(self.edge_index.size(1), dtype=torch.float32)

    def test_temporal_gated_conv_shape(self):
        conv = TemporalGatedConv(in_channels=16, out_channels=32, kernel_size=3)
        # Input shape: [Batch * Nodes, Features, Length]
        x = torch.randn(self.batch_size * self.num_nodes, 16, self.seq_len)
        out = conv(x)
        self.assertEqual(out.shape, (self.batch_size * self.num_nodes, 32, self.seq_len))

    def test_stgcn_forward_pass(self):
        # Input history state tensor: [B, N, T, F]
        x_history = torch.randn(self.batch_size, self.num_nodes, self.seq_len, self.in_features)
        
        organ_hazards, cascade_alert = self.model(x_history, self.edge_index, self.edge_weight)
        
        # Verify output dimensional bounds
        # Organ failure probabilities: [Batch_Size, Num_Nodes]
        self.assertEqual(organ_hazards.shape, (self.batch_size, self.num_nodes))
        # Global systemic cascade alert probability: [Batch_Size, 1]
        self.assertEqual(cascade_alert.shape, (self.batch_size, 1))
        
        # Verify standard Sigmoid output bounds [0.0, 1.0]
        self.assertTrue(torch.all(organ_hazards >= 0.0) and torch.all(organ_hazards <= 1.0))
        self.assertTrue(torch.all(cascade_alert >= 0.0) and torch.all(cascade_alert <= 1.0))

if __name__ == "__main__":
    unittest.main()