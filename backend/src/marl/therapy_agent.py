import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TherapyAgentPolicy(nn.Module):
    """
    Actor-Critic neural network policy for the 'therapy_agent'.
    Simulates clinical treatment policies (IV Fluids, Norepinephrine, Oxygenation, CRRT)
    to stabilize the simulated patient.
    """
    
    def __init__(self, state_dim: int = 14, action_dim: int = 4, hidden_dim: int = 64):
        """
        Args:
            state_dim (int): Input observation size (14)
            action_dim (int): Number of clinical continuous interventions (default: 4)
            hidden_dim (int): Nodes per hidden layer
        """
        super(TherapyAgentPolicy, self).__init__()
        
        # Shared Feature Extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- Actor Head ---
        # Maps hidden features to continuous target clinician doses
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Bounds doses between 0.0 and 1.0 (empty to maximum clinical rates)
        )
        
        # Exploration parameters
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # --- Critic Head ---
        # Estimates expected recovery reward/stabilization potential
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_dict: dict) -> tuple:
        """
        Args:
            obs_dict (dict): Observation components matching environment layout
            
        Returns:
            tuple: (action_means, action_stds, state_values)
        """
        organ_states = torch.as_tensor(obs_dict["organ_states"], dtype=torch.float32)
        vitals = torch.as_tensor(obs_dict["vitals_and_labs"], dtype=torch.float32)
        interventions = torch.as_tensor(obs_dict["active_interventions"], dtype=torch.float32)
        
        # Check batch shape
        if len(organ_states.shape) == 1:
            organ_states = organ_states.unsqueeze(0)
            vitals = vitals.unsqueeze(0)
            interventions = interventions.unsqueeze(0)
            
        state_vector = torch.cat([organ_states, vitals, interventions], dim=-1)
        
        # Compute forward representations
        features = self.shared_net(state_vector)
        
        action_means = self.action_mean(features)
        action_stds = torch.exp(self.action_log_std).expand_as(action_means)
        state_values = self.value_head(features)
        
        return action_means, action_stds, state_values