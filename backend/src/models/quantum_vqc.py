import torch
import torch.nn as nn
import pennylane as qml
import logging
import os
import numpy as np
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumVQCEncoder(nn.Module):
    """
    Hybrid Classical-Quantum State Encoder.
    Fuses 64-dim compressed waveform features and 64-dim clinical lab/vital features
    into a unified 128-dimensional classical state, projects it into an 8-qubit
    Variational Quantum Circuit (VQC) via PennyLane, and extracts a quantum-enhanced
    128-dimensional Patient State Vector.
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3, classical_in_dim: int = 128, patient_state_dim: int = 128):
        """
        Args:
            n_qubits (int): Number of simulated qubits (default: 8).
            n_layers (int): Number of strongly entangling VQC layers (default: 3).
            classical_in_dim (int): Sum of waveform (64) + lab (64) dimensions.
            patient_state_dim (int): Output dimensionality of the unified Patient State Vector (128).
        """
        super(QuantumVQCEncoder, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.classical_in_dim = classical_in_dim
        self.patient_state_dim = patient_state_dim

        # 1. Classical pre-projection layer
        # Maps 128-dim fused features to 24-dim parameters (3 rotation angles per qubit for 8 qubits)
        self.pre_projection = nn.Linear(classical_in_dim, n_qubits * 3)
        self.activation = nn.Tanh()  # Bounds values to [-1, 1] to serve as stable rotation angles (radians)

        # 2. Define the PennyLane device and QNode
        # We use default.qubit simulator with the torch interface for autograd integration
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Define the QNode as a class-internal method wrapped in qml.qnode
        self.qnode = qml.QNode(self._vqc_circuit, self.dev, interface="torch")

        # 3. Variational Weights for Quantum Gates
        # Shape: [n_layers, n_qubits, 3] representing rotation angles for entangling layers
        # Declared as a PyTorch parameter so optimizer can update them during training
        self.vqc_weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 3, dtype=torch.float32) * 0.01
        )

        # 4. Post-Quantum Fusion Layer
        # Concatenates original 128-dim classical features with the 8-dim quantum expectation values
        # and projects them back to the final 128-dimensional Patient State Vector
        self.post_fusion = nn.Linear(classical_in_dim + n_qubits, patient_state_dim)
        self.layer_norm = nn.LayerNorm(patient_state_dim)

    def _vqc_circuit(self, inputs: torch.Tensor, weights: torch.Tensor) -> Tuple[qml.measurements.MeasurementProcess, ...]:
        """
        Core Variational Quantum Circuit mapping classical features to quantum Hilbert space.
        
        Args:
            inputs (Tensor): Angle embeddings of shape [24] (3 angles per qubit)
            weights (Tensor): Learnable gate parameters of shape [n_layers, n_qubits, 3]
        """
        # Step A: State Preparation (Angle Embedding)
        # We apply independent RX, RY, RZ rotations to each qubit based on classical inputs
        for q in range(self.n_qubits):
            offset = q * 3
            qml.RX(inputs[offset], wires=q)
            qml.RY(inputs[offset + 1], wires=q)
            qml.RZ(inputs[offset + 2], wires=q)

        # Step B: Variational Layers (Strongly Entangling Layers)
        # Combines parameterized single-qubit rotations with circular CNOT entangling gates
        # to construct deep non-linear relationships across the state features
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

        # Step C: Measurement
        # Measure expectation values of Pauli-Z operator on all qubits, returning a vector of size 8
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, waveform_emb: torch.Tensor, lab_features: torch.Tensor) -> torch.Tensor:
        """
        Executes the hybrid forward pass.
        
        Args:
            waveform_emb (Tensor): Compressed waveform embedding [Batch_Size, 64]
            lab_features (Tensor): Imputed and aligned clinical lab features [Batch_Size, 64]
            
        Returns:
            Tensor: Hybrid Quantum-Classical Patient State Vector of shape [Batch_Size, 128]
        """
        batch_size = waveform_emb.size(0)
        
        # Step 1: Classical Concatenation
        # Shape: [Batch_Size, 128]
        classical_fused = torch.cat([waveform_emb, lab_features], dim=-1)
        
        # Step 2: Extract bounded rotation parameters
        # Shape: [Batch_Size, 24]
        q_inputs = self.activation(self.pre_projection(classical_fused))
        
        # Step 3: Run inputs through the VQC (batched)
        quantum_outputs = []
        for i in range(batch_size):
            # Evaluate single sample through PennyLane QNode
            # Output shape: [8]
            q_out = self.qnode(q_inputs[i], self.vqc_weights)
            # PennyLane returns a list of tensors; stack them into a single tensor
            quantum_outputs.append(torch.stack(q_out))
            
        # Shape: [Batch_Size, 8]
        # Cast PennyLane outputs from Double (float64) to Float (float32) to match PyTorch weights
        q_features = torch.stack(quantum_outputs).to(dtype=torch.float32)
        
        # Step 4: Post-Quantum Classical Fusion
        # Concat shape: [Batch_Size, 136] (128 classical + 8 quantum features)
        combined = torch.cat([classical_fused, q_features], dim=-1)
        
        # Project back to patient state dimensions with normalization
        # Shape: [Batch_Size, 128]
        patient_state_vector = self.layer_norm(self.post_fusion(combined))
        
        return patient_state_vector

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Hybrid Quantum VQC Encoder ---")
    
    # 1. Initialize your quantum model
    model = QuantumVQCEncoder()
    print("Quantum Model initialized successfully.")

    # 2. Load a small piece of your processed data to use as a test batch
    processed_file = r"C:\Users\tanis\Q-MODS-Engine\backend\data\processed\patient_1001_layout_clean.npy"
    
    if os.path.exists(processed_file):
        raw_signal = np.load(processed_file)
        # Take the first 64 samples of the ECG signal to act as our 64-dim embedding
        # Add a batch dimension so shape becomes [Batch_Size=1, Features=64]
        mock_waveform_emb = torch.tensor(raw_signal[:64, 0], dtype=torch.float32).unsqueeze(0)
        print(f"Loaded processed data. Created waveform tensor with shape: {mock_waveform_emb.shape}")
    else:
        print("Processed data file not found. Using random test numbers instead.")
        mock_waveform_emb = torch.randn(1, 64)

    # 3. Create mock clinical lab features (Shape: [Batch_Size=1, Features=64])
    mock_lab_features = torch.randn(1, 64)

    # 4. Run the forward pass through the Quantum Circuit!
    print("Passing data through the Quantum Circuit (this might take a few seconds)...")
    with torch.no_grad(): # Tells PyTorch we are just testing, not training yet
        patient_state_vector = model(mock_waveform_emb, mock_lab_features)

    print("\n--- SUCCESS! ---")
    print(f"Output Patient State Vector shape: {patient_state_vector.shape}")
    print("Your data successfully traveled through the classical layers, into the qubits, and back!")