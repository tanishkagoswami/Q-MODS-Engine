import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.quantum_vqc import QuantumVQCEncoder

class TestQuantumVQCEncoder(unittest.TestCase):
    """Unit tests for the 8-qubit hybrid classical-quantum state encoder."""

    def setUp(self):
        self.batch_size = 2
        self.n_qubits = 8
        self.n_layers = 3
        
        self.encoder = QuantumVQCEncoder(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            classical_in_dim=128,
            patient_state_dim=128
        )

    def test_vqc_forward_and_gradient(self):
        # Simulated 64-dim waveform embedding + 64-dim clinical lab values
        waveform_emb = torch.randn(self.batch_size, 64, requires_grad=True)
        lab_features = torch.randn(self.batch_size, 64, requires_grad=True)
        
        # Execute forward pass
        patient_state = self.encoder(waveform_emb, lab_features)
        
        # Validate output shape: [Batch_Size, patient_state_dim]
        self.assertEqual(patient_state.shape, (self.batch_size, 128))
        
        # Verify backpropagation can traverse PennyLane quantum circuit to compute gradients
        loss = torch.sum(patient_state)
        loss.backward()
        
        # Check if gradients successfully flow back to classical inputs
        self.assertIsNotNone(waveform_emb.grad)
        self.assertIsNotNone(lab_features.grad)
        self.assertIsNotNone(self.encoder.vqc_weights.grad)

if __name__ == "__main__":
    unittest.main()