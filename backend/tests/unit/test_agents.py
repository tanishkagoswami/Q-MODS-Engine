import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.marl.env_simulator import QMODSMultiAgentEnv

class TestMARLSimulator(unittest.TestCase):
    """Unit tests for Multi-Agent environmental dynamics and cross-talk logic."""

    def setUp(self):
        self.env = QMODSMultiAgentEnv()

    def test_env_initialization(self):
        obs, info = self.env.reset()
        
        # Validate baseline elements are set
        for agent in ["patho_agent", "therapy_agent"]:
            self.assertIn(agent, obs)
            self.assertEqual(obs[agent]["organ_states"].shape, (5,))
            self.assertEqual(obs[agent]["vitals_and_labs"].shape, (5,))
            self.assertEqual(obs[agent]["active_interventions"].shape, (4,))

    def test_step_dynamics_and_rewards(self):
        self.env.reset()
        
        # Configure mild intervention actions
        action_dict = {
            "patho_agent": np.array([0.1, 0.0, 0.0], dtype=np.float32),      # Minor renal stress
            "therapy_agent": np.array([0.2, 0.0, 0.5, 0.0], dtype=np.float32)  # Moderate oxygen + fluids
        }
        
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        
        # Check step indexing
        self.assertEqual(self.env.current_step, 1)
        
        # Verify competitive multi-agent reward signs
        # The pathogen wants failure (positive), while clinician wants stability (negative reward on organ failure)
        self.assertGreater(rewards["patho_agent"], 0.0)
        self.assertLess(rewards["therapy_agent"], 0.0)

    def test_excessive_dose_penalization(self):
        self.env.reset()
        
        # Setup toxic vasopressor infusion rate
        toxic_action = {
            "patho_agent": np.zeros(3, dtype=np.float32),
            "therapy_agent": np.array([0.0, 0.95, 0.0, 0.0], dtype=np.float32) # Extreme vasopressor dose (>0.8)
        }
        
        # Run multiple steps so the smoothed active dose crosses the 0.8 toxicity threshold
        self.env.step(toxic_action)
        self.env.step(toxic_action)
        _, rewards, _, _, _ = self.env.step(toxic_action)
        
        # Toxic vasopressors trigger an extra -5x multiplier on the therapy agent's cumulative reward
        self.assertLess(rewards["therapy_agent"], -0.5)

if __name__ == "__main__":
    unittest.main()