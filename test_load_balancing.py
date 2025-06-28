#!/usr/bin/env python3
"""
Test Financial Load Balancing Kernel
"""

import numpy as np
from pathlib import Path
from max.graph import Graph, TensorType, ops
from max.graph.api import DeviceRef, DType
from max.inference import InferenceSession
from max.tensor import Tensor
from max.devices import CPU, GPU, Accelerator, accelerator_count

def test_financial_load_balancing():
    """Test the financial load balancing kernel independently"""
    print("🧪 Testing Financial Load Balancing Kernel")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    num_experts = 4
    
    print(f"📏 Test dimensions:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Number of experts: {num_experts}")
    
    # Device setup
    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"🚀 Using {device}")
    
    # Custom kernels path
    mojo_kernels = Path(__file__).parent / "kernels"
    
    try:
        # Create graph with load balancing operation
        def load_balancing_graph(
            gate_probs,
            expert_assignments,
            expert_utilization,
            risk_penalty_weight
        ):
            return ops.custom(
                name="financial_load_balancing",
                device=DeviceRef.from_device(device),
                values=[
                    gate_probs,
                    expert_assignments,
                    expert_utilization,
                    risk_penalty_weight
                ],
                out_types=[
                    # aux_loss: [1] - float16
                    TensorType(
                        dtype=DType.float16,
                        shape=[1],
                        device=DeviceRef.from_device(device),
                    ),
                    # risk_metrics: [num_experts] - float16
                    TensorType(
                        dtype=DType.float16,
                        shape=[num_experts],
                        device=DeviceRef.from_device(device),
                    ),
                ],
            )
        
        graph = Graph(
            "financial_load_balancing_test",
            forward=load_balancing_graph,
            input_types=[
                TensorType(DType.float16, [batch_size, seq_len, num_experts], DeviceRef.from_device(device)),
                TensorType(DType.float16, [batch_size, seq_len, num_experts], DeviceRef.from_device(device)),
                TensorType(DType.float16, [num_experts], DeviceRef.from_device(device)),
                TensorType(DType.float16, [1], DeviceRef.from_device(device)),
            ],
            custom_extensions=[mojo_kernels],
        )
        
        # Create session and load model
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Generate test inputs
        np.random.seed(42)
        gate_probs = np.random.rand(batch_size, seq_len, num_experts).astype(np.float16)
        expert_assignments = np.random.rand(batch_size, seq_len, num_experts).astype(np.float16)
        expert_utilization = np.random.rand(num_experts).astype(np.float16)
        risk_penalty_weight = np.array([0.1]).astype(np.float16)
        
        # Normalize gate probabilities
        gate_probs = gate_probs / np.sum(gate_probs, axis=2, keepdims=True)
        
        # Convert to tensors and move to device
        inputs = [
            Tensor.from_numpy(gate_probs).to(device),
            Tensor.from_numpy(expert_assignments).to(device),
            Tensor.from_numpy(expert_utilization).to(device),
            Tensor.from_numpy(risk_penalty_weight).to(device),
        ]
        
        # Run inference
        results = model.execute(*inputs)
        
        # Get results
        aux_loss = results[0].to(CPU()).to_numpy()
        risk_metrics = results[1].to(CPU()).to_numpy()
        
        # Validate outputs
        print("✅ Output validation:")
        print(f"   Auxiliary loss shape: {aux_loss.shape}")
        print(f"   Risk metrics shape: {risk_metrics.shape}")
        print(f"   Auxiliary loss value: {aux_loss[0]:.6f}")
        print(f"   Risk metrics range: [{risk_metrics.min():.3f}, {risk_metrics.max():.3f}]")
        
        # Check output dimensions
        assert aux_loss.shape == (1,)
        assert risk_metrics.shape == (num_experts,)
        
        # Check that auxiliary loss is reasonable (should be positive and small)
        assert aux_loss[0] >= 0
        assert aux_loss[0] < 1.0
        
        # Check that risk metrics are positive
        assert np.all(risk_metrics >= 0)
        
        print("✅ Financial load balancing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Financial load balancing test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_financial_load_balancing()
    exit(0 if success else 1)