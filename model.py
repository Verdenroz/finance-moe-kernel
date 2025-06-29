import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from max.torch import CustomOpLibrary

# Load Mojo kernels
mojo_kernels = Path(__file__).parent / "kernels"
ops = CustomOpLibrary(mojo_kernels)

class FinanceMoEModel(nn.Module):
    def __init__(self, hidden_size=16, num_domains=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_domains = num_domains
        
        # Domain expert router with balanced initialization
        self.domain_router = nn.Linear(hidden_size, num_domains)
        
        # Initialize with balanced biases
        with torch.no_grad():
            self.domain_router.bias.fill_(0.0)
            nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.03)  # Even smaller std
        
        # Domain names for interpretability
        self.domain_names = [
            "Equities", "Fixed Income", "Commodities", 
            "FX", "Derivatives", "Credit"
        ]
        
    def forward(self, sequence_embeddings, market_volatility, risk_factors):
        batch_size, seq_len, hidden_size = sequence_embeddings.shape
        
        # Prepare output tensors
        domain_assignments = torch.zeros(
            (batch_size, seq_len), 
            dtype=torch.int32, 
            device=sequence_embeddings.device
        )
        routing_probs = torch.zeros(
            (batch_size, seq_len, self.num_domains), 
            dtype=torch.float32, 
            device=sequence_embeddings.device
        )
        
        # Get router weights and bias
        router_weight = self.domain_router.weight.detach().t().contiguous()
        router_bias = self.domain_router.bias.detach()
        
        # Call custom Mojo operation
        ops.finance_router(
            domain_assignments,           
            routing_probs,               
            sequence_embeddings.detach().contiguous(), 
            router_weight.contiguous(),               
            router_bias.contiguous(),                 
            market_volatility.detach().contiguous(),  
            risk_factors.detach().contiguous()        
        )
        
        # For gradient computation during training
        if self.training:
            routing_logits = self.domain_router(sequence_embeddings)
            
            # Minimal market condition adjustments for gradient stability
            market_adjustment = market_volatility.unsqueeze(-1) * torch.tensor([
                0.5,    # Equities 
                -1.0,   # Fixed Income 
                0.8,    # Commodities 
                0.6,    # FX 
                1.5,    # Derivatives 
                0.4     # Credit 
            ], device=routing_logits.device) * 0.3
            
            risk_adjustment = risk_factors * torch.tensor([
                0.5,    # Equities 
                -0.8,   # Fixed Income 
                0.6,    # Commodities 
                1.0,    # FX 
                1.2,    # Derivatives 
                0.5     # Credit 
            ], device=routing_logits.device) * 0.3
            
            routing_logits = routing_logits + market_adjustment + risk_adjustment
            routing_probs_diff = torch.softmax(routing_logits / 0.6, dim=-1)  # Match kernel temperature
            
            # Use straight-through estimator
            routing_probs = routing_probs.detach() + routing_probs_diff - routing_probs_diff.detach()
        
        # Simulate expert predictions
        predictions = self._simulate_expert_predictions(
            sequence_embeddings, domain_assignments, routing_probs
        )
        
        return predictions, {
            'domain_assignments': domain_assignments,
            'routing_probs': routing_probs,
            'domain_names': self.domain_names
        }
    
    def _simulate_expert_predictions(self, embeddings, assignments, probs):
        """Simulate expert predictions based on domain assignments"""
        batch_size, seq_len = assignments.shape
        predictions = torch.zeros((batch_size, seq_len, 1), device=embeddings.device)
        
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


def create_advanced_sample_data(batch_size=1, seq_len=30, hidden_size=16, asset_type='mixed', device='cpu'):
    """Create highly distinct sample financial data with clearer asset characteristics"""
    
    if asset_type.lower() == 'equities':
        # Equities: moderate volatility + momentum + avoid cyclical patterns
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.4
        
        # Strong momentum component (key differentiator)
        momentum = torch.cumsum(torch.randn(batch_size, seq_len, 1, device=device) * 0.2, dim=1)
        base_embeddings[:, :, :6] += momentum.expand(batch_size, seq_len, 6)
        
        # Trend consistency (avoid cyclical patterns)
        trend_strength = torch.randn(batch_size, 1, 1, device=device) * 0.25
        base_embeddings[:, :, 6:12] += trend_strength.expand(batch_size, seq_len, 6)
        
        # Avoid high complexity/variance (differentiator from commodities)
        base_embeddings = base_embeddings * 0.8  # Reduce overall complexity
        
        # Target volatility: 1.5-4% (avoid overlap with FX/commodities)
        volatility = torch.rand(batch_size, seq_len, device=device) * 0.025 + 0.015
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.015
        
    elif asset_type.lower() == 'fixed_income':
        # Fixed Income: ultra-low volatility + extreme stability + small values
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.08  # Very small
        
        # Extreme stability - most values should be very small
        stability_mask = torch.rand(batch_size, seq_len, hidden_size, device=device) < 0.8
        base_embeddings = base_embeddings * stability_mask.float() * 0.5
        
        # Mean reversion - values should stay close to zero
        base_embeddings = base_embeddings * 0.3
        
        # Very low volatility: 0.1-0.4%
        volatility = torch.rand(batch_size, seq_len, device=device) * 0.003 + 0.001
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.001
        
    elif asset_type.lower() == 'commodities':
        # Commodities: moderate volatility + high complexity + strong cyclical patterns
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.6
        
        # Very strong cyclical patterns (key differentiator)
        t = torch.linspace(0, 6*3.14159, seq_len, device=device)
        cyclical_1 = torch.sin(t).unsqueeze(0).unsqueeze(-1) * 0.9  # Stronger
        cyclical_2 = torch.cos(t * 1.3).unsqueeze(0).unsqueeze(-1) * 0.7
        cyclical_3 = torch.sin(t * 0.7).unsqueeze(0).unsqueeze(-1) * 0.5
        
        base_embeddings[:, :, :5] += cyclical_1.expand(batch_size, seq_len, 5)
        base_embeddings[:, :, 5:10] += cyclical_2.expand(batch_size, seq_len, 5)
        base_embeddings[:, :, 10:15] += cyclical_3.expand(batch_size, seq_len, 5)
        
        # Supply/demand shocks (high variance spikes)
        shock_mask = torch.rand(batch_size, seq_len, 1, device=device) < 0.15
        shocks = shock_mask.float() * torch.randn(batch_size, seq_len, 1, device=device) * 1.2
        base_embeddings[:, :, :8] += shocks.expand(batch_size, seq_len, 8)
        
        # High complexity measure
        base_embeddings = base_embeddings + torch.sin(base_embeddings) * 0.3
        
        # Target volatility: 1-2.5% (clear separation from FX and Equities)
        volatility = torch.rand(batch_size, seq_len, device=device) * 0.015 + 0.01
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.010
        
    elif asset_type.lower() == 'fx':
        # FX: low-moderate volatility + very strong trends + directional bias
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.25
        
        # Very strong trending behavior (key differentiator from credit/commodities)
        trend = torch.cumsum(torch.randn(batch_size, seq_len, 1, device=device) * 0.18, dim=1)
        base_embeddings[:, :, :10] += trend.expand(batch_size, seq_len, 10)
        
        # Strong directional persistence (carry trade effects)
        direction = torch.randn(batch_size, 1, 1, device=device).sign() * 0.5
        base_embeddings[:, :, 10:] += direction.expand(batch_size, seq_len, hidden_size-10)
        
        # Avoid cyclical patterns (differentiator from commodities)
        # Avoid jump patterns (differentiator from credit)
        base_embeddings = torch.tanh(base_embeddings) * 0.7  # Smooth out extremes
        
        # Target volatility: 0.5-1.3% (clear separation)
        volatility = torch.rand(batch_size, seq_len, device=device) * 0.008 + 0.005
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.020  # Higher risk sensitivity
        
    elif asset_type.lower() == 'derivatives':
        # Derivatives: high volatility + extreme complexity + non-linear patterns
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 1.5
        
        # Strong non-linear patterns (key differentiator)
        base_embeddings = base_embeddings + torch.sin(base_embeddings * 2.5) * 0.8
        base_embeddings = base_embeddings + (base_embeddings ** 2).sign() * (base_embeddings.abs() ** 1.3) * 0.4
        
        # Option-like convexity patterns
        convexity = torch.relu(base_embeddings) ** 1.8 - torch.relu(-base_embeddings) ** 1.8
        base_embeddings[:, :, :8] += convexity[:, :, :8] * 0.4
        
        # Leverage effects and volatility clustering
        vol_clustering = torch.zeros_like(base_embeddings[:, :, :6])
        for i in range(1, seq_len):
            vol_clustering[:, i, :] = vol_clustering[:, i-1, :] * 0.8 + torch.randn(batch_size, 6, device=device) * 0.6
        base_embeddings[:, :, 8:14] += vol_clustering
        
        # Create extreme values (large value ratio)
        extreme_mask = torch.rand(batch_size, seq_len, hidden_size, device=device) < 0.4
        base_embeddings = base_embeddings + extreme_mask.float() * torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.8
        
        # High volatility: 3-7% (clear separation)
        volatility = torch.rand(batch_size, seq_len, device=device) * 0.04 + 0.03
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.050
        
    elif asset_type.lower() == 'credit':
        # Credit: low volatility + jump patterns + tail risk + asymmetric events
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.15
        
        # Baseline stability (most values should be small)
        stable_mask = torch.rand(batch_size, seq_len, hidden_size, device=device) < 0.7
        base_embeddings = base_embeddings * stable_mask.float() * 0.4
        
        # Credit event jumps (key differentiator) - create large value spikes
        jump_prob = 0.12  # 12% chance of credit event
        jump_mask = torch.rand(batch_size, seq_len, 1, device=device) < jump_prob
        jumps = jump_mask.float() * torch.randn(batch_size, seq_len, 1, device=device) * 1.5  # Large jumps
        base_embeddings[:, :, :6] += jumps.expand(batch_size, seq_len, 6)
        
        # Credit spread widening effects (persistent impact)
        spread_widening = torch.zeros(batch_size, seq_len, 1, device=device)
        for i in range(1, seq_len):
            if torch.rand(1) < 0.18:  # 18% chance of spread event
                spread_widening[:, i:, :] += torch.randn(1) * 0.8
        base_embeddings[:, :, 6:12] += spread_widening.expand(batch_size, seq_len, 6)
        
        # Create positive skewness (more positive jumps, key credit characteristic)
        skew_adjustment = torch.abs(torch.randn(batch_size, seq_len, 6, device=device)) * 0.4
        base_embeddings[:, :, :6] += skew_adjustment
        
        # Tail risk patterns (large range but low standard deviation most of the time)
        tail_events = torch.zeros_like(base_embeddings[:, :, :4])
        tail_mask = torch.rand(batch_size, seq_len, 4, device=device) < 0.05  # 5% tail events
        tail_events = tail_mask.float() * torch.randn(batch_size, seq_len, 4, device=device) * 2.0
        base_embeddings[:, :, 12:] += tail_events
        
        # Avoid trending patterns (differentiator from FX)
        # Keep baseline low and stable except for events
        
        # Target volatility: 0.2-0.8% (clear separation from FX)
        volatility = torch.rand(batch_size, seq_len, device=device) * 0.006 + 0.002
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.008
        
    else:  # Mixed/default
        base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.5
        volatility = torch.rand(batch_size, seq_len, device=device) * 0.02 + 0.005
        risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.01
    
    return base_embeddings, volatility, risk_factors


def analyze_advanced_routing_quality(model, asset_types, num_tests=3):
    """Comprehensive analysis with detailed feature analysis"""
    
    print("=" * 90)
    print("ADVANCED ROUTING QUALITY ANALYSIS WITH MULTI-STAGE DECISION TREE")
    print("=" * 90)
    
    overall_accuracy = {}
    overall_confidence = {}
    
    for asset_type in asset_types:
        print(f"\n{'='*25} TESTING {asset_type.upper()} {'='*25}")
        
        accuracies = []
        confidences = []
        feature_analysis = []
        
        for test_run in range(num_tests):
            # Create asset-specific sample data
            embeddings, volatility, risk = create_advanced_sample_data(
                batch_size=1, seq_len=25, asset_type=asset_type
            )
            
            # Analyze embedding characteristics (matching kernel logic)
            embedding_stats = {
                'mean': embeddings[0].mean().item(),
                'std': embeddings[0].std().item(),
                'variance': embeddings[0].var().item(),
                'range': (embeddings[0].max() - embeddings[0].min()).item(),
                'abs_mean': embeddings[0].abs().mean().item(),
                'large_values': (embeddings[0].abs() > 0.5).float().mean().item(),
                'small_values': (embeddings[0].abs() < 0.1).float().mean().item(),
                'positive_ratio': (embeddings[0] > 0).float().mean().item(),
            }
            
            # Calculate derived indicators (matching kernel)
            momentum_strength = abs(embedding_stats['mean'])
            complexity_measure = embedding_stats['variance'] * embedding_stats['range']
            stability_measure = embedding_stats['small_values'] * (1.0 - embedding_stats['large_values'])
            jump_indicator = max(0.0, embedding_stats['large_values'] - 0.3) * embedding_stats['range']
            
            print(f"\nRun {test_run + 1}/{num_tests}:")
            print(f"  Volatility: {volatility.min():.4f} - {volatility.max():.4f} (avg: {volatility.mean():.4f})")
            print(f"  Risk: {risk.min():.4f} - {risk.max():.4f}")
            print(f"  Embedding stats:")
            print(f"    - Complexity measure: {complexity_measure:.3f}")
            print(f"    - Stability measure: {stability_measure:.3f}")
            print(f"    - Jump indicator: {jump_indicator:.3f}")
            print(f"    - Momentum strength: {momentum_strength:.3f}")
            print(f"    - Large values ratio: {embedding_stats['large_values']:.3f}")
            
            # Forward pass
            with torch.no_grad():
                predictions, routing_info = model(embeddings, volatility, risk)
            
            # Analyze routing results
            assignments = routing_info['domain_assignments'][0].numpy()
            probs = routing_info['routing_probs'][0].numpy()
            
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
            print(f"  Accuracy: {accuracy:.1f}% ({correct_assignments}/25)")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Distribution: {domain_counts}")
            
            # Show most confused domains
            if accuracy < 100:
                wrong_assignments = assignments[assignments != expected_domain]
                if len(wrong_assignments) > 0:
                    wrong_domains = np.bincount(wrong_assignments, minlength=6)
                    most_confused = np.argmax(wrong_domains)
                    print(f"  Most confused with: {model.domain_names[most_confused]} ({wrong_domains[most_confused]} times)")
            
            feature_analysis.append({
                'complexity': complexity_measure,
                'stability': stability_measure,
                'jump_indicator': jump_indicator,
                'momentum': momentum_strength,
                'volatility_avg': volatility.mean().item()
            })
        
        # Summary statistics
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_confidence = np.mean(confidences)
        
        overall_accuracy[asset_type] = avg_accuracy
        overall_confidence[asset_type] = avg_confidence
        
        # Feature analysis summary
        avg_features = {k: np.mean([f[k] for f in feature_analysis]) for k in feature_analysis[0].keys()}
        
        print(f"\n📊 SUMMARY FOR {asset_type.upper()}:")
        print(f"   Accuracy: {avg_accuracy:.1f}% ± {std_accuracy:.1f}%")
        print(f"   Confidence: {avg_confidence:.3f}")
        print(f"   Avg Features:")
        print(f"     - Complexity: {avg_features['complexity']:.3f}")
        print(f"     - Stability: {avg_features['stability']:.3f}")
        print(f"     - Jump indicator: {avg_features['jump_indicator']:.3f}")
        print(f"     - Momentum: {avg_features['momentum']:.3f}")
        print(f"     - Volatility: {avg_features['volatility_avg']:.4f}")
        
        # Performance assessment
        if avg_accuracy >= 95:
            print("   ✅ EXCELLENT routing quality")
        elif avg_accuracy >= 85:
            print("   ✔️  VERY GOOD routing quality")
        elif avg_accuracy >= 75:
            print("   ✔️  GOOD routing quality")
        elif avg_accuracy >= 60:
            print("   ⚠️  MODERATE routing quality - needs improvement")
        else:
            print("   ❌ POOR routing quality - major issues")
    
    # Overall performance summary
    print(f"\n{'='*30} OVERALL RESULTS {'='*30}")
    avg_overall_accuracy = np.mean(list(overall_accuracy.values()))
    avg_overall_confidence = np.mean(list(overall_confidence.values()))
    
    print(f"Overall Average Accuracy: {avg_overall_accuracy:.1f}%")
    print(f"Overall Average Confidence: {avg_overall_confidence:.3f}")
    
    # Performance ranking
    sorted_assets = sorted(overall_accuracy.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPerformance Ranking:")
    for rank, (asset, accuracy) in enumerate(sorted_assets, 1):
        print(f"  {rank}. {asset.capitalize()}: {accuracy:.1f}%")
    
    # Identify improvement areas
    excellent_assets = [asset for asset, acc in overall_accuracy.items() if acc >= 95]
    good_assets = [asset for asset, acc in overall_accuracy.items() if 75 <= acc < 95]
    problem_assets = [asset for asset, acc in overall_accuracy.items() if acc < 75]
    
    if excellent_assets:
        print(f"\n✅ Excellent performance (≥95%): {', '.join(excellent_assets)}")
    if good_assets:
        print(f"✔️  Good performance (75-94%): {', '.join(good_assets)}")
    if problem_assets:
        print(f"⚠️  Needs improvement (<75%): {', '.join(problem_assets)}")
    
    return overall_accuracy, overall_confidence


# Test the advanced model
if __name__ == "__main__":
    print("Testing Advanced Finance MoE Model with Multi-Stage Decision Tree...")
    
    # Create model
    model = FinanceMoEModel()
    
    # Test with different asset types
    asset_types = ['equities', 'fixed_income', 'commodities', 'fx', 'derivatives', 'credit']
    
    # Run comprehensive analysis
    accuracy_results, confidence_results = analyze_advanced_routing_quality(
        model, asset_types, num_tests=3
    )