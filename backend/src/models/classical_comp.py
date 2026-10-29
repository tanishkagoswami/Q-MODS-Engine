import torch
import torch.nn as nn
from torch.nn.utils import parametrizations
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChausalDilatedConv1DBlock(nn.Module):
    """
    A single Residual Causal Dilated Convolutional Block for the TCN.
    Uses causal padding to ensure no information leakage from the future.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super(ChausalDilatedConv1DBlock, self).__init__()
        
        # Calculate padding needed to keep output length equal to input length with dilation
        self.padding = (kernel_size - 1) * dilation
        
        # First Dilated Convolutional Layer
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=self.padding, dilation=dilation
        )
        # Weight normalization stabilizes deep TCN training
        self.conv1 = parametrizations.weight_norm(self.conv1)
        self.chomp1 = Chomp1d(self.padding)  # Truncates the trailing padding to force causality
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second Dilated Convolutional Layer
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=self.padding, dilation=dilation
        )
        self.conv2 = parametrizations.weight_norm(self.conv2)
        self.chomp2 = Chomp1d(self.padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Construct the block pipeline
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Downsample residual mapping if channel sizes change
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Helper module to strip out the padding added on the right side of the sequence.
    This guarantees that the convolution remains strictly causal.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class WaveformTCNCompressor(nn.Module):
    """
    1D Temporal Convolutional Network (TCN) to compress multi-channel 
    high-frequency waveforms into a static latent embedding.
    """
    
    def __init__(self, in_channels: int = 3, embed_dim: int = 64, 
                 num_channels: List[int] = [16, 32, 64], kernel_size: int = 3, dropout: float = 0.2):
        """
        Args:
            in_channels (int): Number of input waveform channels (e.g., 3 for ECG, ABP, Respiration).
            embed_dim (int): Dimensionality of the target compressed vector (default: 64).
            num_channels (List[int]): Filter sizes for each hidden layer in the TCN.
            kernel_size (int): Size of the convolutional kernel.
            dropout (float): Dropout probability.
        """
        super(WaveformTCNCompressor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation growth (1, 2, 4, 8, etc.)
            in_c = in_channels if i == 0 else num_channels[i-1]
            out_c = num_channels[i]
            layers.append(
                ChausalDilatedConv1DBlock(
                    in_c, out_c, kernel_size, 
                    dilation=dilation_size, dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*layers)
        
        # Fully connected pooling layer to squeeze sequence length down to a static embedding
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Waveform tensor of shape [Batch_Size, Channels, Sequence_Length]
                              e.g., [32, 3, 2000] (2000 ticks at 125Hz is 16 seconds of data)
                              
        Returns:
            torch.Tensor: Compressed waveform vector of shape [Batch_Size, Embed_Dim]
        """
        # Step 1: Run through causal dilated convolutions
        # Shape: [Batch_Size, num_channels[-1], Sequence_Length]
        features = self.network(x)
        
        # Step 2: Adaptive Global Average Pooling across the temporal dimension
        # Shape: [Batch_Size, num_channels[-1], 1]
        pooled = self.pool(features).squeeze(-1)
        
        # Step 3: Map to the final flat embedding dimension
        # Shape: [Batch_Size, Embed_Dim]
        embedding = self.fc(pooled)
        
        return embedding