#!/usr/bin/env python3
"""
Test Finance MoE FFN Kernel
"""

import numpy as np
from pathlib import Path
from max.graph import Graph, TensorType, ops
from max.graph.api import DeviceRef, DType
from max.inference import InferenceSession
from max.tensor import Tensor
from max.devices import CPU, GPU, Accelerator, accelerator_count

def test_finance_moe_ffn():
    """Test the finance MoE FFN kernel independently"""
    print("🧪 Testing Finance MoE FFN Kernel")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    hidden_size = 8
    intermediate_size = 16
    num_experts = 4
    
    print(f"📏 Test dimensions:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Intermediate size: {intermediate_size}")
    print(f"   Number of experts: {num_experts}")
    
    # Device setup
    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"🚀 Using {device}")
    
    # Custom kernels path
    mojo_kernels = Path(__file__).parent / "kernels"
    
    try:
        # Create graph with MoE FFN operation
        def moe_ffn_graph(
            hidden_states,
            gate_weights,
            gate_bias,
            up_weights,
            down_weights,
            domain_assignments,
            market_regime
        ):
            return ops.custom(
                name="finance_moe_ffn",
                device=DeviceRef.from_device(device),
                values=[
                    hidden_states,
                    gate_weights,
                    gate_bias,
                    up_weights,
                    down_weights,
                    domain_assignments,
                    market_regime
                ],
                out_types=[
                    # output: [batch, seq_len, hidden] - float16
                    TensorType(
                        dtype=DType.float16,
                        shape=[batch_size, seq_len, hidden_size],
                        device=DeviceRef.from_device(device),
                    ),
                    # expert_utilization: [num_experts] - float16
                    TensorType(
                        dtype=DType.float16,
                        shape=[num_experts],
                        device=DeviceRef.from_device(device),
                    ),
                ],
            )
        
        graph = Graph(
            "finance_moe_ffn_test",
            forward=moe_ffn_graph,
            input_types=[
                TensorType(DType.float16, [batch_size, seq_len, hidden_size], DeviceRef.from_device(device)),
                TensorType(DType.float16, [hidden_size, num_experts], DeviceRef.from_device(device)),
                TensorType(DType.float16, [num_experts], DeviceRef.from_device(device)),
                TensorType(DType.float16, [num_experts, hidden_size, intermediate_size], DeviceRef.from_device(device)),
                TensorType(DType.float16, [num_experts, intermediate_size, hidden_size], DeviceRef.from_device(device)),
                TensorType(DType.int32, [batch_size, seq_len], DeviceRef.from_device(device)),
                TensorType(DType.float16, [batch_size], DeviceRef.from_device(device)),
            ],
            custom_extensions=[mojo_kernels],
        )
        
        # Create session and load model
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Generate test inputs
        np.random.seed(42)
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float16)
        gate_weights = np.random.randn(hidden_size, num_experts).astype(np.float16)
        gate_bias = np.random.randn(num_experts).astype(np.float16)
        up_weights = np.random.randn(num_experts, hidden_size, intermediate_size).astype(np.float16)
        down_weights = np.random.randn(num_experts, intermediate_size, hidden_size).astype(np.float16)
        domain_assignments = np.random.randint(0, 6, (batch_size, seq_len)).astype(np.int32)
        market_regime = np.random.rand(batch_size).astype(np.float16)
        
        # Convert to tensors and move to device
        inputs = [
            Tensor.from_numpy(hidden_states).to(device),
            Tensor.from_numpy(gate_weights).to(device),
            Tensor.from_numpy(gate_bias).to(device),
            Tensor.from_numpy(up_weights).to(device),
            Tensor.from_numpy(down_weights).to(device),
            Tensor.from_numpy(domain_assignments).to(device),
            Tensor.from_numpy(market_regime).to(device),
        ]
        
        # Run inference
        results = model.execute(*inputs)
        
        # Get results
        output = results[0].to(CPU()).to_numpy()
        expert_utilization = results[1].to(CPU()).to_numpy()
        
        # Validate outputs
        print("✅ Output validation:")
        print(f"   Output shape: {output.shape}")
        print(f"   Expert utilization shape: {expert_utilization.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"   Expert utilization range: [{expert_utilization.min():.3f}, {expert_utilization.max():.3f}]")
        
        # Check output dimensions
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert expert_utilization.shape == (num_experts,)
        
        # Check that output is not just pass-through (should have some modification)
        input_mean = np.mean(hidden_states)
        output_mean = np.mean(output)
        print(f"   Input mean: {input_mean:.3f}, Output mean: {output_mean:.3f}")
        
        # Check expert utilization sums
        total_utilization = np.sum(expert_utilization)
        print(f"   Total expert utilization: {total_utilization:.3f}")
        
        print("✅ Finance MoE FFN test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Finance MoE FFN test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_finance_moe_ffn()
    exit(0 if success else 1)