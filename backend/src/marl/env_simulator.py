import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import logging
from typing import Tuple, Dict as TypeDict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QMODSMultiAgentEnv(gym.Env):
    """
    Custom Multi-Agent Reinforcement Learning (MARL) Environment for ICU Interventions.
    Simulates physiology transitions of 5 interconnected organ systems.
    
    Agents:
      - 'patho_agent': Simulates pathogenic/disease progression stressors.
      - 'therapy_agent': Simulates therapeutic actions (Fluids, Vasopressors, Ventilation, etc.).
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: TypeDict[str, Any] = None):
        super(QMODSMultiAgentEnv, self).__init__()
        
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 24)  # Represents a 24-hour simulation window
        self.current_step = 0
        
        # 5 Organs: Renal, Pulmonary, Cardiovascular, Hepatic, CNS
        self.num_organs = 5
        
        # --- Observation Space ---
        obs_space = Dict({
            "organ_states": Box(low=0.0, high=1.0, shape=(self.num_organs,), dtype=np.float32),
            "vitals_and_labs": Box(low=0.0, high=2.0, shape=(5,), dtype=np.float32),
            "active_interventions": Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        
        self.observation_space = Dict({
            "patho_agent": obs_space,
            "therapy_agent": obs_space
        })
        
        # --- Action Space ---
        self.action_space_patho = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space_therapy = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.action_space = Dict({
            "patho_agent": self.action_space_patho,
            "therapy_agent": self.action_space_therapy
        })
        
        self.reset()

    def reset(self, seed: int = None, options: TypeDict[str, Any] = None) -> Tuple[TypeDict[str, Any], TypeDict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        
        self.organ_states = np.random.uniform(0.05, 0.20, size=(self.num_organs,)).astype(np.float32)
        self.vitals_and_labs = np.ones((5,), dtype=np.float32)
        self.active_interventions = np.zeros((4,), dtype=np.float32)
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> TypeDict[str, Any]:
        """Formats the shared environment state observation for both agents."""
        shared_state = {
            "organ_states": self.organ_states.copy(),
            "vitals_and_labs": self.vitals_and_labs.copy(),
            "active_interventions": self.active_interventions.copy()
        }
        return {
            "patho_agent": shared_state,
            "therapy_agent": shared_state
        }

    def step(self, action_dict: TypeDict[str, np.ndarray]) -> Tuple[TypeDict[str, Any], TypeDict[str, float], TypeDict[str, bool], TypeDict[str, bool], TypeDict[str, Any]]:
        self.current_step += 1
        
        patho_act = action_dict.get("patho_agent", np.zeros(3, dtype=np.float32))
        therapy_act = action_dict.get("therapy_agent", np.zeros(4, dtype=np.float32))
        
        # TASK 3.4 FIX: Clip actions to prevent negative doses or overflow exploits
        patho_act = np.clip(patho_act, 0.0, 1.0)
        therapy_act = np.clip(therapy_act, 0.0, 1.0)
        
        # Update active interventions with smoothing
        self.active_interventions = 0.5 * self.active_interventions + 0.5 * therapy_act
        
        fluids, vasopressor, oxygen, crrt = self.active_interventions
        renal_insult, pulmonary_insult, cardiac_insult = patho_act
        prev_states = self.organ_states.copy()
        
        # A. Cardiovascular Node
        cardio_damage = (0.15 * cardiac_insult + 0.08 * prev_states[0] + 0.05 * prev_states[3])
        cardio_assist = 0.12 * vasopressor + 0.05 * fluids
        self.organ_states[2] = np.clip(prev_states[2] + cardio_damage - cardio_assist, 0.0, 1.0)
        
        # B. Renal Node
        renal_damage = (0.18 * renal_insult + 0.15 * prev_states[2] + 0.05 * prev_states[3])
        renal_assist = 0.15 * crrt + 0.04 * fluids
        if vasopressor > 0.7:
            renal_damage += 0.08 * (vasopressor - 0.7)
        self.organ_states[0] = np.clip(prev_states[0] + renal_damage - renal_assist, 0.0, 1.0)
        
        # C. Pulmonary Node
        pulm_damage = 0.20 * pulmonary_insult + 0.10 * prev_states[2]
        if fluids > 0.6:
            pulm_damage += 0.12 * (fluids - 0.6)
        pulm_assist = 0.15 * oxygen
        self.organ_states[1] = np.clip(prev_states[1] + pulm_damage - pulm_assist, 0.0, 1.0)
        
        # D. Hepatic Node
        hep_damage = 0.10 * prev_states[2] + 0.08 * prev_states[0]
        self.organ_states[3] = np.clip(prev_states[3] + hep_damage - 0.02, 0.0, 1.0)
        
        # E. CNS Node
        cns_damage = 0.10 * prev_states[0] + 0.10 * prev_states[3] + 0.10 * prev_states[1]
        cns_assist = 0.04 * vasopressor + 0.04 * oxygen
        self.organ_states[4] = np.clip(prev_states[4] + cns_damage - cns_assist, 0.0, 1.0)
        
        # Update Lab/Vital Proxies
        self.vitals_and_labs[0] = np.clip(1.0 + 0.4 * vasopressor + 0.2 * fluids - 0.7 * self.organ_states[2], 0.1, 2.0)
        self.vitals_and_labs[1] = np.clip(1.0 + 0.1 * oxygen - 0.5 * self.organ_states[1] - 0.2 * self.organ_states[2], 0.0, 1.2)
        self.vitals_and_labs[2] = np.clip(1.0 + 1.5 * self.organ_states[0] - 0.4 * crrt, 0.2, 3.0)
        self.vitals_and_labs[3] = np.clip(1.0 + 1.5 * self.organ_states[3], 0.2, 3.0)
        self.vitals_and_labs[4] = np.clip(1.0 - 0.8 * self.organ_states[4], 0.1, 1.0)
        
        # Compute Rewards
        total_failure_index = np.mean(self.organ_states)
        
        # TASK 3.3 FIX: Use the requested dose (therapy_act) for toxicity penalties, not the smoothed dose
        req_fluids = therapy_act[0]
        req_vaso = therapy_act[1]
        
        therapy_toxicity_penalty = 0.0
        if req_vaso > 0.8:
            therapy_toxicity_penalty += 0.15 * (req_vaso - 0.8)
        if req_fluids > 0.8:
            therapy_toxicity_penalty += 0.10 * (req_fluids - 0.8)
            
        reward_patho = float(total_failure_index * 10.0)
        reward_therapy = float(-total_failure_index * 10.0 - therapy_toxicity_penalty * 5.0)
        
        rewards = {
            "patho_agent": reward_patho,
            "therapy_agent": reward_therapy
        }
        
        patient_dead = bool(np.any(self.organ_states >= 0.98) or self.vitals_and_labs[0] < 0.2)
        terminated = {
            "patho_agent": patient_dead,
            "therapy_agent": patient_dead,
            "__all__": patient_dead
        }
        
        reached_limit = bool(self.current_step >= self.max_steps)
        truncated = {
            "patho_agent": reached_limit,
            "therapy_agent": reached_limit,
            "__all__": reached_limit
        }
        
        obs = self._get_obs()
        return obs, rewards, terminated, truncated, {}