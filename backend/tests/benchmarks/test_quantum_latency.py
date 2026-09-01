import time
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.quantum_vqc import QuantumVQCEncoder

def run_performance_benchmark():
    print("=" * 60)
    print("STARTING LATENCY BENCHMARK: HYBRID VQC vs CLASSICAL NETWORK")
    print("=" * 60)
    
    batch_size = 1
    iterations = 20
    
    # 1. Initialize Hybrid Quantum Encoder
    quantum_encoder = QuantumVQCEncoder(n_qubits=8, n_layers=3)
    
    # 2. Initialize an equivalent Classical linear feature projector
    classical_projector = nn.Sequential(
        nn.Linear(128, 136),
        nn.ReLU(),
        nn.Linear(136, 128)
    )
    
    # Prepare dummy tensors
    wave_emb = torch.randn(batch_size, 64)
    lab_feats = torch.randn(batch_size, 64)
    fused_classical = torch.cat([wave_emb, lab_feats], dim=-1)
    
    # Benchmark purely classical network
    print("\nWarmup classical network...")
    for _ in range(5):
        _ = classical_projector(fused_classical)
        
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = classical_projector(fused_classical)
    classical_duration = (time.perf_counter() - start_time) / iterations * 1000
    
    print(f"Classical inference execution latency: {classical_duration:.4f} ms")
    
    # Benchmark hybrid PennyLane VQC simulation
    print("\nWarmup hybrid Quantum VQC simulator...")
    for _ in range(3):
        _ = quantum_encoder(wave_emb, lab_feats)
        
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = quantum_encoder(wave_emb, lab_feats)
    quantum_duration = (time.perf_counter() - start_time) / iterations * 1000
    
    print(f"Hybrid Quantum VQC execution latency:   {quantum_duration:.4f} ms")
    print("-" * 60)
    print(f"Latency Overhead Multiplier: {quantum_duration / classical_duration:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    run_performance_benchmark()