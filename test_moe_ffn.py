#!/usr/bin/env python3
"""
Test Finance MoE FFN Kernel
"""

import numpy as np
from pathlib import Path
from max.graph import Graph, TensorType, ops, DeviceRef
from max.engine import InferenceSession
from max.dtype import DType
from max.driver import CPU, Accelerator, accelerator_count, Tensor

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
    
    # Device setup - force CPU
    device = CPU()
    print(f"🚀 Using {device}")
    
    # Custom kernels path
    mojo_kernels = Path(__file__).parent / "kernels"
    
    try:
        print("🔧 Setting up graph...")
        
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
            print("📊 Creating custom operation...")
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
                    # output: [batch, seq_len, hidden] - float32
                    TensorType(
                        dtype=DType.float32,
                        shape=[batch_size, seq_len, hidden_size],
                        device=DeviceRef.from_device(device),
                    ),
                    # expert_utilization: [num_experts] - float32
                    TensorType(
                        dtype=DType.float32,
                        shape=[num_experts],
                        device=DeviceRef.from_device(device),
                    ),
                ],
            )
        
        print("🔗 Building graph...")
        graph = Graph(
            "finance_moe_ffn_test",
            forward=moe_ffn_graph,
            input_types=[
                TensorType(DType.float32, [batch_size, seq_len, hidden_size], DeviceRef.from_device(device)),  # hidden_states
                TensorType(DType.float32, [num_experts, hidden_size], DeviceRef.from_device(device)),          # gate_weights  
                TensorType(DType.float32, [num_experts], DeviceRef.from_device(device)),                       # gate_bias
                TensorType(DType.float32, [num_experts, hidden_size, intermediate_size], DeviceRef.from_device(device)),  # up_weights
                TensorType(DType.float32, [num_experts, intermediate_size, hidden_size], DeviceRef.from_device(device)),  # down_weights
                TensorType(DType.int32, [batch_size, seq_len], DeviceRef.from_device(device)),                 # domain_assignments
                TensorType(DType.float32, [1], DeviceRef.from_device(device)),                                 # market_regime
            ],
            custom_extensions=[mojo_kernels],
        )
        print("✅ Graph created successfully")
        
        # Create session and load model
        print("🎯 Creating inference session...")
        session = InferenceSession(devices=[device])
        print("📦 Loading model...")
        model = session.load(graph)
        print("✅ Model loaded successfully")
        
        # Generate test inputs
        print("🎲 Generating test inputs...")
        np.random.seed(42)
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        gate_weights = np.random.randn(num_experts, hidden_size).astype(np.float32)  # Fixed shape
        gate_bias = np.random.randn(num_experts).astype(np.float32)
        up_weights = np.random.randn(num_experts, hidden_size, intermediate_size).astype(np.float32)
        down_weights = np.random.randn(num_experts, intermediate_size, hidden_size).astype(np.float32)
        domain_assignments = np.random.randint(0, num_experts, (batch_size, seq_len)).astype(np.int32)
        market_regime = np.random.rand(1).astype(np.float32)  # Fixed shape
        
        print("📤 Converting inputs to tensors and moving to device...")
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
        print("✅ Inputs prepared successfully")
        
        # Run inference
        print("🚀 Running inference...")
        results = model.execute(*inputs)
        print("✅ Inference completed successfully")
        
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
        
        print("🎉 Finance MoE FFN test passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Finance MoE FFN test failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_finance_moe_ffn()
    exit(0 if success else 1)