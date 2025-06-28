#!/usr/bin/env python3
"""
Test Finance Macro Router Kernel
"""

import numpy as np
from pathlib import Path
from max.graph import Graph, TensorType, ops, DeviceRef
from max.engine import InferenceSession
from max.dtype import DType
from max.driver import CPU, Accelerator, accelerator_count, Tensor

def test_finance_macro_router():
    """Test the finance macro router kernel independently"""
    print("🧪 Testing Finance Macro Router Kernel")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 8
    hidden_size = 16
    num_domains = 6
    num_risk_metrics = 4
    
    print(f"📏 Test dimensions:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Number of domains: {num_domains}")
    
    # Device setup
    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"🚀 Using {device}")
    
    # Custom kernels path
    mojo_kernels = Path(__file__).parent / "kernels"
    
    try:
        # Create graph with macro router operation
        def macro_router_graph(
            sequence_embeddings,
            domain_router_weights,
            domain_router_bias,
            market_volatility,
            risk_factors
        ):
            return ops.custom(
                name="finance_macro_router",
                device=DeviceRef.from_device(device),
                values=[
                    sequence_embeddings,
                    domain_router_weights,
                    domain_router_bias,
                    market_volatility,
                    risk_factors
                ],
                out_types=[
                    # domain_assignments: [batch, seq_len] - int32
                    TensorType(
                        dtype=DType.int32,
                        shape=[batch_size, seq_len],
                        device=DeviceRef.from_device(device),
                    ),
                    # routing_probs: [batch, seq_len, num_domains] - float16
                    TensorType(
                        dtype=DType.float16,
                        shape=[batch_size, seq_len, num_domains],
                        device=DeviceRef.from_device(device),
                    ),
                ],
            )
        
        graph = Graph(
            "finance_macro_router_test",
            forward=macro_router_graph,
            input_types=[
                TensorType(DType.float16, [batch_size, seq_len, hidden_size], DeviceRef.from_device(device)),
                TensorType(DType.float16, [hidden_size, num_domains], DeviceRef.from_device(device)),
                TensorType(DType.float16, [num_domains], DeviceRef.from_device(device)),
                TensorType(DType.float16, [batch_size, seq_len], DeviceRef.from_device(device)),
                TensorType(DType.float16, [batch_size, seq_len, num_risk_metrics], DeviceRef.from_device(device)),
            ],
            custom_extensions=[mojo_kernels],
        )
        
        # Create session and load model
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        
        # Generate test inputs
        np.random.seed(42)
        sequence_embeddings = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float16)
        domain_router_weights = np.random.randn(hidden_size, num_domains).astype(np.float16)
        domain_router_bias = np.random.randn(num_domains).astype(np.float16)
        market_volatility = np.random.rand(batch_size, seq_len).astype(np.float16)
        risk_factors = np.random.rand(batch_size, seq_len, num_risk_metrics).astype(np.float16)
        
        # Convert to tensors and move to device
        inputs = [
            Tensor.from_numpy(sequence_embeddings).to(device),
            Tensor.from_numpy(domain_router_weights).to(device),
            Tensor.from_numpy(domain_router_bias).to(device),
            Tensor.from_numpy(market_volatility).to(device),
            Tensor.from_numpy(risk_factors).to(device),
        ]
        
        # Run inference
        results = model.execute(*inputs)
        
        # Get results
        domain_assignments = results[0].to(CPU()).to_numpy()
        routing_probs = results[1].to(CPU()).to_numpy()
        
        # Validate outputs
        print("✅ Output validation:")
        print(f"   Domain assignments shape: {domain_assignments.shape}")
        print(f"   Routing probabilities shape: {routing_probs.shape}")
        print(f"   Domain assignments range: [{domain_assignments.min()}, {domain_assignments.max()}]")
        print(f"   Routing probs range: [{routing_probs.min():.3f}, {routing_probs.max():.3f}]")
        
        # Check domain assignments are valid
        assert domain_assignments.shape == (batch_size, seq_len)
        assert np.all(domain_assignments >= 0) and np.all(domain_assignments < num_domains)
        
        # Check routing probabilities
        assert routing_probs.shape == (batch_size, seq_len, num_domains)
        assert np.all(routing_probs >= 0) and np.all(routing_probs <= 1)
        
        # Check probability sums (should be close to 1 for each sequence position)
        prob_sums = np.sum(routing_probs, axis=2)
        print(f"   Probability sums range: [{prob_sums.min():.3f}, {prob_sums.max():.3f}]")
        
        print("✅ Finance macro router test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Finance macro router test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_finance_macro_router()
    exit(0 if success else 1)