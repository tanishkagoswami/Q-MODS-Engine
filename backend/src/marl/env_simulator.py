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
        # The observation consists of:
        # 1. Current organ health states (values between 0.0 [fully healthy] and 1.0 [completely failed])
        # 2. Vitals / Lab proxies [MAP, SpO2, Creatinine, Bilirubin, GCS]
        # 3. Active drug/intervention doses (normalized [0.0, 1.0])
        obs_space = Dict({
            "organ_states": Box(low=0.0, high=1.0, shape=(self.num_organs,), dtype=np.float32),
            "vitals_and_labs": Box(low=0.0, high=2.0, shape=(5,), dtype=np.float32),
            "active_interventions": Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        
        # Ray RLlib Multi-Agent standard defines spaces mapped per agent ID
        self.observation_space = Dict({
            "patho_agent": obs_space,
            "therapy_agent": obs_space
        })
        
        # --- Action Space ---
        # 1. Patho Agent (Continuous multi-dimensional disease stressor vector)
        #    - Action index 0: Renal septic insult
        #    - Action index 1: Pulmonary inflammatory insult (ARDS)
        #    - Action index 2: Cardiogenic stress / hypoxia
        self.action_space_patho = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 2. Therapy Agent (Continuous clinician interventions)
        #    - Action index 0: IV Fluids rate
        #    - Action index 1: Vasopressor (Norepinephrine) titration
        #    - Action index 2: Fraction of Inspired Oxygen (FiO2) / PEEP pressure
        #    - Action index 3: Renal Replacement (CRRT) / Dialysis intensity
        self.action_space_therapy = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.action_space = Dict({
            "patho_agent": self.action_space_patho,
            "therapy_agent": self.action_space_therapy
        })
        
        # Initialize internal physiological states
        self.reset()

    def reset(self, seed: int = None, options: TypeDict[str, Any] = None) -> Tuple[TypeDict[str, Any], TypeDict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # Seed initial organ failure metrics between [0.05, 0.2] (mildly unstable ICU admission state)
        self.organ_states = np.random.uniform(0.05, 0.20, size=(self.num_organs,)).astype(np.float32)
        
        # Initialize normal physiologic parameters
        # Indices: [MAP, SpO2, Creatinine, Bilirubin, GCS] (scaled relative to healthy normal baseline of 1.0)
        self.vitals_and_labs = np.ones((5,), dtype=np.float32)
        
        # Current active drug infusions
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
        """
        Processes simultaneous actions from both agents and steps the physiological system forward.
        """
        self.current_step += 1
        
        patho_act = action_dict.get("patho_agent", np.zeros(3, dtype=np.float32))
        therapy_act = action_dict.get("therapy_agent", np.zeros(4, dtype=np.float32))
        
        # --- 1. Physiological State Transitions (Differential Cross-Talk Model) ---
        # Update active interventions with smoothing (pharmacokinetics representation)
        self.active_interventions = 0.5 * self.active_interventions + 0.5 * therapy_act
        
        # Extract interventions
        fluids, vasopressor, oxygen, crrt = self.active_interventions
        
        # Extract pathogenic actions
        renal_insult, pulmonary_insult, cardiac_insult = patho_act
        
        # Track cross-talk values
        prev_states = self.organ_states.copy()
        
        # A. Cardiovascular Node (Index 2)
        # Degraded by: Cardiac insult, Renal fluid overload (pulmonary backup), and Hepatic sepsis
        # Stabilized by: Vasopressors (up to a limit) and fluids (hemodynamic volume)
        cardio_damage = (0.15 * cardiac_insult + 
                         0.08 * prev_states[0] +  # Renal failure -> Fluid volume stress
                         0.05 * prev_states[3])   # Hepatic failure -> Toxic state
        cardio_assist = 0.12 * vasopressor + 0.05 * fluids
        self.organ_states[2] = np.clip(prev_states[2] + cardio_damage - cardio_assist, 0.0, 1.0)
        
        # B. Renal Node (Index 0) - Highly perfusion-sensitive!
        # Degraded by: Renal insult, cardiovascular failure (poor perfusion)
        # Stabilized by: CRRT, moderate MAP (balance of fluids & vasopressors)
        renal_damage = (0.18 * renal_insult + 
                        0.15 * prev_states[2] +   # Cardiorenal syndrome (heart failure -> kidney ischemia)
                        0.05 * prev_states[3])    # Hepatorenal syndrome (liver toxin backup)
        renal_assist = 0.15 * crrt + 0.04 * fluids
        # Toxic vasopressor side-effect check: excess vasoconstriction hurts kidneys
        if vasopressor > 0.7:
            renal_damage += 0.08 * (vasopressor - 0.7)
        self.organ_states[0] = np.clip(prev_states[0] + renal_damage - renal_assist, 0.0, 1.0)
        
        # C. Pulmonary Node (Index 1)
        # Degraded by: Pulmonary insult, cardiovascular congestion (pulmonary backup from high fluids)
        # Stabilized by: Oxygen support
        pulm_damage = 0.20 * pulmonary_insult + 0.10 * prev_states[2]
        if fluids > 0.6:  # Fluid overload pooling in lungs
            pulm_damage += 0.12 * (fluids - 0.6)
        pulm_assist = 0.15 * oxygen
        self.organ_states[1] = np.clip(prev_states[1] + pulm_damage - pulm_assist, 0.0, 1.0)
        
        # D. Hepatic Node (Index 3)
        # Degraded by: Poor systemic perfusion (Cardiovascular failure)
        # Stabilized by: Natural metabolic recovery (slow)
        hep_damage = 0.10 * prev_states[2] + 0.08 * prev_states[0]
        self.organ_states[3] = np.clip(prev_states[3] + hep_damage - 0.02, 0.0, 1.0)
        
        # E. CNS Node (Index 4)
        # Degraded by: Multi-system sepsis cascade (Renal, Pulmonary, Hepatic encephalopathy)
        # Stabilized by: Overall hemodynamic support (Vasopressors/Oxygen preserving perfusion)
        cns_damage = 0.10 * prev_states[0] + 0.10 * prev_states[3] + 0.10 * prev_states[1]
        cns_assist = 0.04 * vasopressor + 0.04 * oxygen
        self.organ_states[4] = np.clip(prev_states[4] + cns_damage - cns_assist, 0.0, 1.0)
        
        # --- 2. Update Lab/Vital Proxies ---
        # MAP (Blood pressure) -> Boosted by vasopressors and fluids; collapsed by cardiovascular failure
        self.vitals_and_labs[0] = np.clip(1.0 + 0.4 * vasopressor + 0.2 * fluids - 0.7 * self.organ_states[2], 0.1, 2.0)
        # SpO2 (Oxygenation) -> Driven by pulmonary function and cardiac output
        self.vitals_and_labs[1] = np.clip(1.0 + 0.1 * oxygen - 0.5 * self.organ_states[1] - 0.2 * self.organ_states[2], 0.0, 1.2)
        # Creatinine (Kidney metric) -> Rises as kidney state worsens
        self.vitals_and_labs[2] = np.clip(1.0 + 1.5 * self.organ_states[0] - 0.4 * crrt, 0.2, 3.0)
        # Bilirubin (Liver metric) -> Rises as liver state worsens
        self.vitals_and_labs[3] = np.clip(1.0 + 1.5 * self.organ_states[3], 0.2, 3.0)
        # GCS (Brain responsiveness) -> Drops as brain state worsens
        self.vitals_and_labs[4] = np.clip(1.0 - 0.8 * self.organ_states[4], 0.1, 1.0)
        
        # --- 3. Compute Multi-Agent Rewards (Zero-Sum + Cooperative Dynamics) ---
        # Base failure cost: mean of overall organ failure states
        total_failure_index = np.mean(self.organ_states)
        
        # Penalty for therapy agent violating safety parameters (fluid overload, toxic pressor doses)
        therapy_toxicity_penalty = 0.0
        if vasopressor > 0.8:
            therapy_toxicity_penalty += 0.15 * (vasopressor - 0.8)
        if fluids > 0.8:
            therapy_toxicity_penalty += 0.10 * (fluids - 0.8)
            
        # Collaborative & Competitive Reward Assignment:
        # - patho_agent wants to MAXIMIZE patient organ failure
        # - therapy_agent wants to MINIMIZE patient organ failure & prevent intervention toxicity
        reward_patho = float(total_failure_index * 10.0)
        reward_therapy = float(-total_failure_index * 10.0 - therapy_toxicity_penalty * 5.0)
        
        rewards = {
            "patho_agent": reward_patho,
            "therapy_agent": reward_therapy
        }
        
        # --- 4. Terminate / Truncate Flags ---
        # The episode terminates if any critical organ is completely destroyed (state reaches 1.0)
        # or if systemic MAP drops to incompatible-with-life limits.
        patient_dead = bool(np.any(self.organ_states >= 0.98) or self.vitals_and_labs[0] < 0.2)
        
        terminated = {
            "patho_agent": patient_dead,
            "therapy_agent": patient_dead,
            "__all__": patient_dead
        }
        
        # Truncate if maximum ICU step horizon is reached
        reached_limit = bool(self.current_step >= self.max_steps)
        truncated = {
            "patho_agent": reached_limit,
            "therapy_agent": reached_limit,
            "__all__": reached_limit
        }
        
        obs = self._get_obs()
        
        return obs, rewards, terminated, truncated, {}