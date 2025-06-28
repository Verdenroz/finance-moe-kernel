#!/usr/bin/env python3
"""
Test Market Regime Detector Kernel
"""

import numpy as np
from pathlib import Path
from max.graph import Graph, TensorType, ops, DeviceRef
from max.engine import InferenceSession
from max.dtype import DType
from max.driver import CPU, Accelerator, accelerator_count, Tensor

def test_market_regime_detector():
    """Test the market regime detector kernel independently"""
    print("🧪 Testing Market Regime Detector Kernel")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    num_features = 4
    lookback_window = 5
    
    print(f"📏 Test dimensions:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Number of features: {num_features}")
    print(f"   Lookback window: {lookback_window}")
    
    # Device setup - force CPU
    device = CPU()
    print(f"🚀 Using {device}")
    
    # Custom kernels path
    mojo_kernels = Path(__file__).parent / "kernels"
    
    try:
        # Create graph with market regime detector operation
        def market_regime_graph(price_sequences, lookback_window_tensor):
            return ops.custom(
                name="market_regime_detector",
                device=DeviceRef.from_device(device),
                values=[
                    price_sequences,
                    lookback_window_tensor
                ],
                out_types=[
                    # regime_indicators: [batch, 3] - bull/bear/sideways probabilities
                    TensorType(
                        dtype=DType.float32,
                        shape=[batch_size, 3],
                        device=DeviceRef.from_device(device),
                    ),
                    # volatility_indicators: [batch, 1] - volatility level
                    TensorType(
                        dtype=DType.float32,
                        shape=[batch_size, 1],
                        device=DeviceRef.from_device(device),
                    ),
                ],
            )
        
        print("🔧 Creating market regime detector graph")
        graph = Graph(
            "market_regime_detector_test",
            forward=market_regime_graph,
            input_types=[
                TensorType(DType.float32, [batch_size, seq_len, num_features], DeviceRef.from_device(device)),
                TensorType(DType.int32, [1], DeviceRef.from_device(device)),
            ],
            custom_extensions=[mojo_kernels],
        )
        print("🔧 Graph created successfully")
        
        # Create session and load model
        session = InferenceSession(devices=[device])
        model = session.load(graph)
        print("🔗 Model loaded successfully")
        # Generate test inputs - simulate price data
        np.random.seed(42)
        
        # Create realistic price sequences
        price_sequences = np.zeros((batch_size, seq_len, num_features), dtype=np.float32)
        
        for b in range(batch_size):
            # Start with base price
            base_price = 100.0
            price_sequences[b, 0, 0] = base_price
            
            # Generate price series with trend and noise
            trend = 0.01 if b == 0 else -0.005  # Bull vs bear market
            
            for t in range(1, seq_len):
                # Price movement with trend and random walk
                change = np.random.normal(trend, 0.02)
                price_sequences[b, t, 0] = price_sequences[b, t-1, 0] * (1 + change)
                
                # Fill other features with random data
                price_sequences[b, t, 1:] = np.random.randn(num_features - 1) * 0.1
        
        lookback_window_tensor = np.array([lookback_window], dtype=np.int32)
        
        print(f"📊 Generated price data:")
        print(f"   Batch 0 price range: [{price_sequences[0, :, 0].min():.2f}, {price_sequences[0, :, 0].max():.2f}]")
        print(f"   Batch 1 price range: [{price_sequences[1, :, 0].min():.2f}, {price_sequences[1, :, 0].max():.2f}]")
        
        # Convert to tensors and move to device
        inputs = [
            Tensor.from_numpy(price_sequences).to(device),
            Tensor.from_numpy(lookback_window_tensor).to(device),
        ]
        
        # Run inference
        results = model.execute(*inputs)
        
        # Get results
        regime_indicators = results[0].to(CPU()).to_numpy()
        volatility_indicators = results[1].to(CPU()).to_numpy()
        
        # Validate outputs
        print("✅ Output validation:")
        print(f"   Regime indicators shape: {regime_indicators.shape}")
        print(f"   Volatility indicators shape: {volatility_indicators.shape}")
        
        for b in range(batch_size):
            bull_prob, bear_prob, sideways_prob = regime_indicators[b]
            volatility = volatility_indicators[b, 0]
            
            print(f"   Batch {b} - Bull: {bull_prob:.3f}, Bear: {bear_prob:.3f}, Sideways: {sideways_prob:.3f}")
            print(f"   Batch {b} - Volatility: {volatility:.3f}")
        
        # Check output dimensions
        assert regime_indicators.shape == (batch_size, 3)
        assert volatility_indicators.shape == (batch_size, 1)
        
        # Check that probabilities are valid (between 0 and 1)
        assert np.all(regime_indicators >= 0) and np.all(regime_indicators <= 1)
        assert np.all(volatility_indicators >= 0)
        
        # Check that regime probabilities sum approximately to 1
        prob_sums = np.sum(regime_indicators, axis=1)
        print(f"   Regime probability sums: {prob_sums}")
        assert np.allclose(prob_sums, 1.0, atol=0.1)
        
        print("✅ Market regime detector test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Market regime detector test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_market_regime_detector()
    exit(0 if success else 1)