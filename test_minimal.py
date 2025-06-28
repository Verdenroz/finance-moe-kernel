#!/usr/bin/env python3
"""
Minimal test to isolate segfault issues
"""

import numpy as np
from pathlib import Path
from max.graph import Graph, TensorType, ops, DeviceRef
from max.engine import InferenceSession
from max.dtype import DType
from max.driver import CPU, Accelerator, accelerator_count, Tensor

def test_minimal():
    """Test minimal case"""
    print("🧪 Minimal Finance Kernel Test")
    
    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"Using device: {device}")
    mojo_kernels = Path(__file__).parent / "kernels"
    
    try:
        print("📦 Creating graph...")
        
        def simple_graph(seq_emb, weights, bias, volatility, risk):
            return ops.custom(
                name="finance_macro_router",
                device=DeviceRef.from_device(device),
                values=[seq_emb, weights, bias, volatility, risk],
                out_types=[
                    TensorType(DType.int32, [1, 1], DeviceRef.from_device(device)),
                    TensorType(DType.float16, [1, 1, 6], DeviceRef.from_device(device)),
                ],
            )
        
        graph = Graph(
            "minimal_test",
            forward=simple_graph,
            input_types=[
                TensorType(DType.float16, [1, 1, 4], DeviceRef.from_device(device)),
                TensorType(DType.float16, [4, 6], DeviceRef.from_device(device)),
                TensorType(DType.float16, [6], DeviceRef.from_device(device)),
                TensorType(DType.float16, [1, 1], DeviceRef.from_device(device)),
                TensorType(DType.float16, [1, 1, 4], DeviceRef.from_device(device)),
            ],
            custom_extensions=[mojo_kernels],
        )
        
        print("✅ Graph created successfully")
        
        print("🚀 Creating session...")
        session = InferenceSession(devices=[device])
        
        print("📥 Loading model...")
        model = session.load(graph)
        
        print("✅ Model loaded successfully")
        
        # Create minimal inputs
        seq_emb = Tensor.from_numpy(np.ones((1, 1, 4), dtype=np.float16)).to(device)
        weights = Tensor.from_numpy(np.ones((4, 6), dtype=np.float16)).to(device)
        bias = Tensor.from_numpy(np.ones(6, dtype=np.float16)).to(device)
        volatility = Tensor.from_numpy(np.ones((1, 1), dtype=np.float16)).to(device)
        risk = Tensor.from_numpy(np.ones((1, 1, 4), dtype=np.float16)).to(device)
        
        print("🎯 Executing model...")
        results = model.execute(seq_emb, weights, bias, volatility, risk)
        
        print("✅ Model executed successfully!")
        print(f"   Result 0 shape: {results[0].shape}")
        print(f"   Result 1 shape: {results[1].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_minimal()
    exit(0 if success else 1)