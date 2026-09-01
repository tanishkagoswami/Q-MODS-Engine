import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PathoAgentPolicy(nn.Module):
    """
    Actor-Critic neural network policy for the 'patho_agent'.
    Simulates disease progression by outputting continuous stressor vectors 
    (Renal septic insult, Pulmonary inflammatory insult, Cardiogenic stress).
    """
    
    def __init__(self, state_dim: int = 14, action_dim: int = 3, hidden_dim: int = 64):
        """
        Args:
            state_dim (int): Input observation size. 
                             (5 organs + 5 labs/vitals + 4 active interventions = 14)
            action_dim (int): Insult/stressor dimensions (default: 3)
            hidden_dim (int): Nodes per hidden layer
        """
        super(PathoAgentPolicy, self).__init__()
        
        # Shared Feature Extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- Actor Head (Action generation) ---
        # Outputs continuous action mean values (mu)
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Bounds actions strictly between 0.0 and 1.0 (env action limits)
        )
        
        # Learnable log standard deviation for exploration in continuous action spaces
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # --- Critic Head (State Value estimation) ---
        # Estimates expected total future cumulative failure progression (V)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_dict: dict) -> tuple:
        """
        Processes multi-dictionary observation format and computes action distribution parameters.
        
        Args:
            obs_dict (dict): Dictionary with keys "organ_states", "vitals_and_labs", "active_interventions"
            
        Returns:
            tuple: (action_means, action_stds, state_values)
        """
        # Flatten and concatenate dictionary observations into a single state vector
        organ_states = torch.as_tensor(obs_dict["organ_states"], dtype=torch.float32)
        vitals = torch.as_tensor(obs_dict["vitals_and_labs"], dtype=torch.float32)
        interventions = torch.as_tensor(obs_dict["active_interventions"], dtype=torch.float32)
        
        # Support batching dimension
        if len(organ_states.shape) == 1:
            organ_states = organ_states.unsqueeze(0)
            vitals = vitals.unsqueeze(0)
            interventions = interventions.unsqueeze(0)
            
        state_vector = torch.cat([organ_states, vitals, interventions], dim=-1)
        
        # Feature extraction
        features = self.shared_net(state_vector)
        
        # Compute Actor and Critic outputs
        action_means = self.action_mean(features)
        action_stds = torch.exp(self.action_log_std).expand_as(action_means)
        state_values = self.value_head(features)
        
        return action_means, action_stds, state_values