import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from max.torch import CustomOpLibrary


mojo_kernels = Path(__file__).parent / "kernels"
ops = CustomOpLibrary(mojo_kernels)

class FinanceMoEModel(nn.Module):
    def __init__(self, hidden_size=16, num_domains=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_domains = num_domains
        
        # Domain expert router with balanced initialization
        self.domain_router = nn.Linear(hidden_size, num_domains)
        
        # Initialize with balanced biases for better GPU performance
        with torch.no_grad():
            self.domain_router.bias.fill_(0.0)
            nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.02)
        
        # Domain names for interpretability
        self.domain_names = [
            "Equities", "Fixed Income", "Commodities", 
            "FX", "Derivatives", "Credit"
        ]
        
        # GPU support
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.cuda()
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  Using CPU (GPU not available)")
        
    def forward(self, sequence_embeddings, market_volatility, risk_factors):
        print(f"🔍 [DEBUG] Forward pass started")
        batch_size, seq_len, hidden_size = sequence_embeddings.shape
        print(f"🔍 [DEBUG] Input shapes - batch_size: {batch_size}, seq_len: {seq_len}, hidden_size: {hidden_size}")
        
        # Move to appropriate device and ensure contiguous memory
        print(f"🔍 [DEBUG] Moving tensors to device: {self.device}")
        sequence_embeddings = sequence_embeddings.to(self.device).contiguous()
        market_volatility = market_volatility.to(self.device).contiguous()
        risk_factors = risk_factors.to(self.device).contiguous()
        print(f"🔍 [DEBUG] Tensors moved to device successfully")
        
        # Validate input shapes and types
        print(f"🔍 [DEBUG] Validating inputs...")
        print(f"  sequence_embeddings: {sequence_embeddings.shape}, dtype: {sequence_embeddings.dtype}")
        print(f"  market_volatility: {market_volatility.shape}, dtype: {market_volatility.dtype}")
        print(f"  risk_factors: {risk_factors.shape}, dtype: {risk_factors.dtype}")
        
        # Check for NaN or Inf values
        if torch.isnan(sequence_embeddings).any():
            print("⚠️ [DEBUG] WARNING: NaN detected in sequence_embeddings")
        if torch.isinf(sequence_embeddings).any():
            print("⚠️ [DEBUG] WARNING: Inf detected in sequence_embeddings")
            
        # Prepare output tensors
        print(f"🔍 [DEBUG] Preparing output tensors...")
        domain_assignments = torch.zeros(
            (batch_size, seq_len), 
            dtype=torch.int32, 
            device=self.device
        ).contiguous()
        routing_probs = torch.zeros(
            (batch_size, seq_len, self.num_domains), 
            dtype=torch.float32, 
            device=self.device
        ).contiguous()
        print(f"🔍 [DEBUG] Output tensors created successfully")
        
        # Get router weights and bias
        print(f"🔍 [DEBUG] Preparing router parameters...")
        router_weight = self.domain_router.weight.detach().t().contiguous()
        router_bias = self.domain_router.bias.detach().contiguous()
        
        print(f"  router_weight: {router_weight.shape}, dtype: {router_weight.dtype}")
        print(f"  router_bias: {router_bias.shape}, dtype: {router_bias.dtype}")
        print(f"🔍 [DEBUG] Router parameters prepared successfully")
        
        # Check if ops library is loaded
        print(f"🔍 [DEBUG] Checking Mojo ops library...")
        try:
            print(f"  ops available: {hasattr(ops, 'finance_router')}")
            if hasattr(ops, 'finance_router'):
                print(f"  finance_router function found")
            else:
                print("❌ [ERROR] finance_router function not found in ops")
                raise AttributeError("finance_router not found in ops")
        except Exception as e:
            print(f"❌ [ERROR] Failed to access ops library: {e}")
            raise
        
        # Call custom Mojo operation with error handling
        print(f"🔍 [DEBUG] Calling Mojo kernel...")
        try:
            print(f"🔍 [DEBUG] About to call ops.finance_router...")
            ops.finance_router(
                domain_assignments,           
                routing_probs,               
                sequence_embeddings.detach(), 
                router_weight,               
                router_bias,                 
                market_volatility.detach(),  
                risk_factors.detach()        
            )
            print(f"✅ [DEBUG] Mojo kernel execution completed successfully!")
            
        except Exception as e:
            print(f"❌ [ERROR] Mojo kernel failed: {type(e).__name__}: {e}")
            print(f"🔍 [DEBUG] Falling back to PyTorch implementation...")
            
            # Fallback to PyTorch implementation
            routing_logits = self.domain_router(sequence_embeddings)
            routing_probs = torch.softmax(routing_logits, dim=-1)
            domain_assignments = torch.argmax(routing_probs, dim=-1).to(torch.int32)
            print(f"✅ [DEBUG] PyTorch fallback completed")
        
        # Validate outputs
        print(f"🔍 [DEBUG] Validating outputs...")
        print(f"  domain_assignments: {domain_assignments.shape}, dtype: {domain_assignments.dtype}")
        print(f"  routing_probs: {routing_probs.shape}, dtype: {routing_probs.dtype}")
        
        # Check output ranges
        print(f"  domain_assignments range: [{domain_assignments.min().item()}, {domain_assignments.max().item()}]")
        print(f"  routing_probs range: [{routing_probs.min().item():.4f}, {routing_probs.max().item():.4f}]")
        print(f"  routing_probs sum per timestep: {routing_probs.sum(dim=-1).mean().item():.4f}")

        # For gradient computation during training
        if self.training:
            routing_logits = self.domain_router(sequence_embeddings)
            
            # Market condition adjustments for gradient stability
            market_adjustment = market_volatility.unsqueeze(-1) * torch.tensor([
                0.5,    # Equities 
                -1.0,   # Fixed Income 
                0.8,    # Commodities 
                0.6,    # FX 
                1.5,    # Derivatives 
                0.4     # Credit 
            ], device=self.device) * 0.3
            
            risk_adjustment = risk_factors * torch.tensor([
                0.5,    # Equities 
                -0.8,   # Fixed Income 
                0.6,    # Commodities 
                1.0,    # FX 
                1.2,    # Derivatives 
                0.5     # Credit 
            ], device=self.device) * 0.3
            
            routing_logits = routing_logits + market_adjustment + risk_adjustment
            routing_probs_diff = torch.softmax(routing_logits / 1.2, dim=-1)
            
            # Use straight-through estimator
            routing_probs = routing_probs.detach() + routing_probs_diff - routing_probs_diff.detach()
        
        # Simulate expert predictions
        print(f"🔍 [DEBUG] Simulating expert predictions...")
        predictions = self._simulate_expert_predictions(
            sequence_embeddings, domain_assignments, routing_probs
        )
        print(f"✅ [DEBUG] Expert predictions completed")
        print(f"  predictions: {predictions.shape}, dtype: {predictions.dtype}")
        
        print(f"✅ [DEBUG] Forward pass completed successfully")
        return predictions, {
            'domain_assignments': domain_assignments,
            'routing_probs': routing_probs,
            'domain_names': self.domain_names
        }
    
    def _simulate_expert_predictions(self, embeddings, assignments, probs):
        """Simulate expert predictions based on domain assignments"""
        batch_size, seq_len = assignments.shape
        predictions = torch.zeros((batch_size, seq_len, 1), device=self.device)
        
        for domain_id in range(self.num_domains):
            domain_mask = (assignments == domain_id).float().unsqueeze(-1)
            
            if domain_id == 0:  # Equities
                momentum = embeddings[:, :, :4].mean(-1, keepdim=True)
                volatility_factor = embeddings.std(-1, keepdim=True)
                domain_pred = torch.tanh(momentum) * (1.0 + volatility_factor)
            elif domain_id == 1:  # Fixed Income
                mean_level = embeddings.mean(-1, keepdim=True)
                domain_pred = torch.sigmoid(mean_level) * 0.3 - 0.15
            elif domain_id == 2:  # Commodities
                trend = embeddings[:, :, :6].mean(-1, keepdim=True)
                cycle = torch.sin(embeddings[:, :, 6:10].mean(-1, keepdim=True) * 3.14159)
                domain_pred = trend * 0.8 + cycle * 0.4
            elif domain_id == 3:  # FX
                trend = embeddings[:, :, :8].mean(-1, keepdim=True)
                noise = torch.randn_like(trend) * 0.05
                domain_pred = torch.tanh(trend) * 0.9 + noise
            elif domain_id == 4:  # Derivatives
                nonlinear = torch.relu(embeddings.mean(-1, keepdim=True)) ** 1.2
                volatility_boost = embeddings.std(-1, keepdim=True) * 2.5
                domain_pred = nonlinear + volatility_boost - 0.5
            else:  # Credit
                base = torch.sigmoid(embeddings.mean(-1, keepdim=True)) * 0.4
                credit_risk = torch.tanh(embeddings.std(-1, keepdim=True)) * 0.2
                domain_pred = base + credit_risk
            
            predictions += domain_pred * domain_mask
        
        return predictions
    

def create_enhanced_sample_data(batch_size=1, seq_len=30, hidden_size=16, asset_type='mixed', device='cpu'):
    """Create highly distinct sample financial data with enhanced characteristics"""
    
    if asset_type.lower() == 'equities':
        # Equities: moderate volatility (1.5-4%) + strong momentum + directional trends
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.3
        
        # Strong momentum component (key differentiator for equities)
        momentum = torch.cumsum(torch.randn(batch_size, seq_len, 1, device=device) * 0.25, dim=1)
        base_embeddings[:, :, :8] += momentum.expand(-1, -1, 8)
        
        # Add trend consistency 
        trend_strength = torch.randn(batch_size, 1, 1, device=device) * 0.3
        base_embeddings[:, :, 8:] += trend_strength.expand(-1, seq_len, hidden_size-8)
        
        # Target volatility: 1.8-3.5% (clear separation from other domains)
        volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.017 + 0.018
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.012
        
    elif asset_type.lower() == 'fixed_income':
        # Fixed Income: ultra-low volatility (<0.5%) + extreme stability
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.06
        
        # Extreme stability - values should be very small and consistent
        stability_mask = torch.rand(batch_size, seq_len, hidden_size, device=device) < 0.85
        base_embeddings = base_embeddings * stability_mask.float() * 0.4
        
        # Mean reversion - keep values close to zero
        base_embeddings = base_embeddings * 0.25
        
        # Ultra-low volatility: 0.1-0.4%
        volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.003 + 0.001
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.002
        
    elif asset_type.lower() == 'commodities':
        # Commodities: moderate volatility (1-2.5%) + strong cyclical patterns + high complexity
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.5
        
        # Very strong cyclical patterns (key differentiator)
        t = torch.linspace(0, 8*3.14159, seq_len, device=device)
        cyclical_1 = torch.sin(t).unsqueeze(0).unsqueeze(-1) * 1.2
        cyclical_2 = torch.cos(t * 1.4).unsqueeze(0).unsqueeze(-1) * 0.9
        cyclical_3 = torch.sin(t * 0.6).unsqueeze(0).unsqueeze(-1) * 0.7
        
        base_embeddings[:, :, :5] += cyclical_1.expand(-1, -1, 5)
        base_embeddings[:, :, 5:10] += cyclical_2.expand(-1, -1, 5)
        base_embeddings[:, :, 10:15] += cyclical_3.expand(-1, -1, 5)
        
        # Supply/demand shocks for complexity
        shock_mask = torch.rand(batch_size, seq_len, 1, device=device) < 0.2
        shocks = shock_mask.float() * torch.randn(batch_size, seq_len, 1, device=device) * 1.5
        base_embeddings[:, :, :6] += shocks.expand(-1, -1, 6)
        
        # High variance target (>0.4 for commodity detection)
        base_embeddings = base_embeddings + torch.sin(base_embeddings) * 0.4
        
        # Target volatility: 1.2-2.2%
        volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.010 + 0.012
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.008
        
    elif asset_type.lower() == 'fx':
        # FX: low-moderate volatility (0.5-1.3%) + extremely strong trends + high directional bias
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.15
        
        # Extremely strong trending behavior (key differentiator from credit/commodities)
        trend = torch.cumsum(torch.randn(batch_size, seq_len, 1, device=device) * 0.35, dim=1)
        base_embeddings[:, :, :14] += trend.expand(-1, -1, 14)
        
        # Very strong directional persistence (carry trade effects)
        direction = torch.randn(batch_size, 1, 1, device=device).sign() * 1.2
        base_embeddings[:, :, 8:] += direction.expand(-1, seq_len, hidden_size-8)
        
        # Add momentum acceleration to distinguish from commodities
        momentum = torch.zeros_like(trend)
        for i in range(1, seq_len):
            momentum[:, i, :] = momentum[:, i-1, :] * 0.95 + torch.randn(batch_size, 1, device=device) * 0.4
        base_embeddings[:, :, :6] += momentum.expand(-1, -1, 6)
        
        # Ensure very high absolute mean for trend detection (much higher than credit)
        base_embeddings = base_embeddings * 1.8
        
        # Keep variance moderate (lower than commodities/credit)
        base_embeddings = base_embeddings + base_embeddings.mean(dim=-1, keepdim=True) * 0.3
        
        # Target volatility: 0.6-1.2%
        volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.006 + 0.006
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.040
        
    elif asset_type.lower() == 'derivatives':
        # Derivatives: high volatility (>3%) + extreme complexity + non-linear patterns
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 1.8
        
        # Strong non-linear patterns
        base_embeddings = base_embeddings + torch.sin(base_embeddings * 3.0) * 1.0
        base_embeddings = base_embeddings + (base_embeddings ** 2).sign() * (base_embeddings.abs() ** 1.5) * 0.5
        
        # Option-like convexity patterns
        convexity = torch.relu(base_embeddings) ** 2.0 - torch.relu(-base_embeddings) ** 2.0
        base_embeddings[:, :, :8] += convexity[:, :, :8] * 0.6
        
        # Volatility clustering
        vol_clustering = torch.zeros_like(base_embeddings[:, :, :8])
        for i in range(1, seq_len):
            vol_clustering[:, i, :] = vol_clustering[:, i-1, :] * 0.9 + torch.randn(batch_size, 8, device=device) * 0.8
        base_embeddings[:, :, 8:] += vol_clustering
        
        # High volatility: 3.2-6%
        volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.028 + 0.032
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.060
        
    elif asset_type.lower() == 'credit':
        # Credit: low volatility (0.2-0.8%) + jump patterns + tail risk + distinct variance profile
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.15
        
        # Baseline stability (most values small) but less than fixed income
        stable_mask = torch.rand(batch_size, seq_len, hidden_size, device=device) < 0.70
        base_embeddings = base_embeddings * stable_mask.float() * 0.4
        
        # Credit event jumps (key differentiator) - more frequent and distinctive
        jump_prob = 0.25
        jump_mask = torch.rand(batch_size, seq_len, 1, device=device) < jump_prob
        jumps = jump_mask.float() * torch.randn(batch_size, seq_len, 1, device=device) * 3.0
        base_embeddings[:, :, :10] += jumps.expand(-1, -1, 10)
        
        # Credit spread widening (persistent impact) - make it more distinct
        spread_widening = torch.zeros(batch_size, seq_len, 1, device=device)
        for i in range(1, seq_len):
            if torch.rand(1) < 0.30:
                spread_widening[:, i:, :] += torch.randn(1, device=device) * 1.5
        base_embeddings[:, :, 8:] += spread_widening.expand(-1, -1, hidden_size-8)
        
        # Credit-specific variance pattern - higher variance than FX but lower than commodities
        variance_pattern = torch.randn(batch_size, seq_len, 4, device=device) * 1.2
        base_embeddings[:, :, 12:16] += variance_pattern
        
        # Make mean close to zero but with occasional spikes (different from FX trending)
        base_embeddings = base_embeddings - base_embeddings.mean(dim=-1, keepdim=True) * 0.7
        
        # Target volatility: 0.3-0.7% (higher than fixed income, lower than FX)
        volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.004 + 0.003
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.010
        
    else:  # Mixed/default
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.5
        volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.02 + 0.005
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.01
    
    return base_embeddings, volatility, risk_factors


def analyze_routing_quality(model, asset_types, num_tests=5, seq_len=25):
    """Comprehensive analysis with enhanced accuracy testing"""
    
    print("=" * 80)
    print("FINANCE MOE ROUTING QUALITY ANALYSIS")
    print("=" * 80)
    
    overall_accuracy = {}
    overall_confidence = {}
    detailed_results = {}
    
    for asset_type in asset_types:
        print(f"\n{'='*20} TESTING {asset_type.upper()} {'='*20}")
        
        accuracies = []
        confidences = []
        
        for test_run in range(num_tests):
            # Create enhanced asset-specific sample data
            embeddings, volatility, risk = create_enhanced_sample_data(
                batch_size=1, seq_len=seq_len, asset_type=asset_type, device=model.device
            )
            
            # Analyze embedding characteristics
            emb_mean = embeddings[0].mean().item()
            emb_std = embeddings[0].std().item()
            emb_var = embeddings[0].var().item()
            emb_abs_mean = embeddings[0].abs().mean().item()
            vol_avg = volatility.mean().item()
            risk_avg = risk.mean().item()
            
            print(f"\nRun {test_run + 1}/{num_tests}:")
            print(f"  Volatility: {vol_avg:.4f} ({'✓' if asset_type == 'equities' and 0.015 <= vol_avg <= 0.04 else '✓' if asset_type == 'fixed_income' and vol_avg < 0.005 else '✓' if asset_type == 'derivatives' and vol_avg > 0.03 else '✓' if asset_type == 'fx' and 0.005 <= vol_avg <= 0.013 else '✓' if asset_type == 'commodities' and 0.01 <= vol_avg <= 0.025 else '✓' if asset_type == 'credit' and 0.002 <= vol_avg <= 0.008 else '⚠'})")
            print(f"  Embedding variance: {emb_var:.3f}")
            print(f"  Embedding abs mean: {emb_abs_mean:.3f}")
            
            # Forward pass
            with torch.no_grad():
                predictions, routing_info = model(embeddings, volatility, risk)
            
            # Analyze routing results
            assignments = routing_info['domain_assignments'][0].cpu().numpy()
            probs = routing_info['routing_probs'][0].cpu().numpy()
            
            # Expected domain mapping
            expected_domain = {
                'equities': 0, 'fixed_income': 1, 'commodities': 2,
                'fx': 3, 'derivatives': 4, 'credit': 5
            }[asset_type]
            
            # Calculate accuracy
            correct_assignments = (assignments == expected_domain).sum()
            accuracy = correct_assignments / len(assignments) * 100
            accuracies.append(accuracy)
            
            # Calculate confidence
            confidence = np.max(probs, axis=1).mean()
            confidences.append(confidence)
            
            # Detailed breakdown
            domain_counts = np.bincount(assignments, minlength=6)
            print(f"  Accuracy: {accuracy:.1f}% ({correct_assignments}/{seq_len})")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Distribution: {domain_counts}")
            
            # Show routing errors
            if accuracy < 100:
                wrong_assignments = assignments[assignments != expected_domain]
                if len(wrong_assignments) > 0:
                    wrong_domains = np.bincount(wrong_assignments, minlength=6)
                    most_confused = np.argmax(wrong_domains)
                    print(f"  Most confused with: {model.domain_names[most_confused]} ({wrong_domains[most_confused]} times)")
        
        # Summary statistics
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_confidence = np.mean(confidences)
        
        overall_accuracy[asset_type] = avg_accuracy
        overall_confidence[asset_type] = avg_confidence
        detailed_results[asset_type] = {
            'accuracies': accuracies,
            'confidences': confidences,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_confidence': avg_confidence
        }
        
        print(f"\n📊 SUMMARY FOR {asset_type.upper()}:")
        print(f"   Accuracy: {avg_accuracy:.1f}% ± {std_accuracy:.1f}%")
        print(f"   Confidence: {avg_confidence:.3f}")
        
        # Performance assessment
        if avg_accuracy >= 95:
            print("   🏆 EXCELLENT routing quality")
        elif avg_accuracy >= 85:
            print("   ✅ VERY GOOD routing quality")
        elif avg_accuracy >= 75:
            print("   ✔️  GOOD routing quality")
        elif avg_accuracy >= 60:
            print("   ⚠️  MODERATE routing quality - needs improvement")
        else:
            print("   ❌ POOR routing quality - major issues")
    
    # Overall performance summary
    print(f"\n{'='*25} OVERALL RESULTS {'='*25}")
    avg_overall_accuracy = np.mean(list(overall_accuracy.values()))
    avg_overall_confidence = np.mean(list(overall_confidence.values()))
    
    print(f"Overall Average Accuracy: {avg_overall_accuracy:.1f}%")
    print(f"Overall Average Confidence: {avg_overall_confidence:.3f}")
    
    # Performance ranking
    sorted_assets = sorted(overall_accuracy.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPerformance Ranking:")
    for rank, (asset, accuracy) in enumerate(sorted_assets, 1):
        print(f"  {rank}. {asset.capitalize()}: {accuracy:.1f}%")
    
    return overall_accuracy, overall_confidence, detailed_results


# Test the model
if __name__ == "__main__":
    print("🧪 Testing Finance MoE Model with GPU Support...")
    print("=" * 60)
    
    # Create model
    model = FinanceMoEModel()
    
    # Test with different asset types
    asset_types = ['equities', 'fixed_income', 'commodities', 'fx', 'derivatives', 'credit']
    
    # Run comprehensive analysis
    accuracy_results, confidence_results, detailed_results = analyze_routing_quality(
        model, asset_types, num_tests=5, seq_len=30
    )
    
    # Check if we've achieved the target
    avg_accuracy = np.mean(list(accuracy_results.values()))
    print(f"\n🎯 TARGET CHECK:")
    print(f"   Current Overall Accuracy: {avg_accuracy:.1f}%")
    print(f"   Target: 85%")
    
    if avg_accuracy >= 85:
        print("   ✅ TARGET ACHIEVED! 🎉")
    else:
        print(f"   ❌ Need {85 - avg_accuracy:.1f}% improvement")
        
        # Show which domains need work
        problem_domains = [asset for asset, acc in accuracy_results.items() if acc < 80]
        if problem_domains:
            print(f"   Domains needing improvement: {', '.join(problem_domains)}")
    
    print(f"\nDevice used: {model.device}")
    if model.device == 'cuda':
        print(f"GPU Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")