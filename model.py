import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from max.torch import CustomOpLibrary

mojo_kernels = Path(__file__).parent / "kernels"
ops = CustomOpLibrary(mojo_kernels)


class FinanceMoEModel(nn.Module):
    def __init__(self, hidden_size=16, num_domains=6):
        """Initializes the FinanceMoEModel.

        Args:
            hidden_size (int): The size of the hidden embeddings.
            num_domains (int): The number of expert domains.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_domains = num_domains

        # Ensure GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this model. Please ensure CUDA is available.")

        self.device = 'cuda'

        # Domain expert router with balanced initialization
        self.domain_router = nn.Linear(hidden_size, num_domains)

        with torch.no_grad():
            self.domain_router.bias.fill_(0.0)
            nn.init.normal_(self.domain_router.weight, mean=0.0, std=0.02)

        # Domain names for interpretability
        self.domain_names = [
            "Equities", "Fixed Income", "Commodities",
            "FX", "Derivatives", "Credit"
        ]

        # Move model to GPU
        self.cuda()

    def forward(self, sequence_embeddings, market_volatility, risk_factors):
        """Performs the forward pass of the model.

        Args:
            sequence_embeddings (torch.Tensor): Input tensor of shape 
                (batch_size, seq_len, hidden_size).
            market_volatility (torch.Tensor): Market volatility tensor of shape
                (batch_size, seq_len, 1).
            risk_factors (torch.Tensor): Risk factors tensor of shape
                (batch_size, seq_len, 1).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The model predictions.
                - dict: A dictionary with routing information including
                  'domain_assignments', 'routing_probs', and 'domain_names'.
        """
        batch_size, seq_len, hidden_size = sequence_embeddings.shape

        # Move inputs to GPU and ensure contiguous memory
        sequence_embeddings = sequence_embeddings.to(self.device).contiguous()
        market_volatility = market_volatility.to(self.device).contiguous()
        risk_factors = risk_factors.to(self.device).contiguous()

        # Validate inputs for NaN/Inf
        if torch.isnan(sequence_embeddings).any() or torch.isinf(sequence_embeddings).any():
            raise ValueError("Invalid values detected in sequence_embeddings")

        # Prepare output tensors
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

        # Get router parameters
        router_weight = self.domain_router.weight.detach().t().contiguous()
        router_bias = self.domain_router.bias.detach().contiguous()

        # Call custom Mojo operation
        if not hasattr(ops, 'finance_router'):
            raise AttributeError("finance_router not found in Mojo ops library")

        ops.finance_router(
            domain_assignments,
            routing_probs,
            sequence_embeddings.detach(),
            router_weight,
            router_bias,
            market_volatility.detach(),
            risk_factors.detach()
        )

        # For gradient computation during training
        if self.training:
            routing_logits = self.domain_router(sequence_embeddings)

            # Market condition adjustments for gradient stability
            market_adjustment = market_volatility.unsqueeze(-1) * torch.tensor([
                0.5, -1.0, 0.8, 0.6, 1.5, 0.4  # Equities, Fixed Income, Commodities, FX, Derivatives, Credit
            ], device=self.device) * 0.3

            risk_adjustment = risk_factors * torch.tensor([
                0.5, -0.8, 0.6, 1.0, 1.2, 0.5  # Equities, Fixed Income, Commodities, FX, Derivatives, Credit
            ], device=self.device) * 0.3

            routing_logits = routing_logits + market_adjustment + risk_adjustment
            routing_probs_diff = torch.softmax(routing_logits / 1.2, dim=-1)

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
        """Simulates expert predictions based on domain assignments.

        Args:
            embeddings (torch.Tensor): The input embeddings.
            assignments (torch.Tensor): The domain assignments for each token.
            probs (torch.Tensor): The routing probabilities for each token.

        Returns:
            torch.Tensor: The simulated expert predictions.
        """
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


def _create_equities_data(batch_size, seq_len, hidden_size, device):
    """Generates sample data for Equities."""
    base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.3
    momentum = torch.cumsum(torch.randn(batch_size, seq_len, 1, device=device) * 0.25, dim=1)
    base_embeddings[:, :, :8] += momentum.expand(-1, -1, 8)
    trend_strength = torch.randn(batch_size, 1, 1, device=device) * 0.3
    base_embeddings[:, :, 8:] += trend_strength.expand(-1, seq_len, hidden_size - 8)
    volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.017 + 0.018
    risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.012
    return base_embeddings, volatility, risk_factors


def _create_fixed_income_data(batch_size, seq_len, hidden_size, device):
    """Generates sample data for Fixed Income."""
    base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.06
    stability_mask = torch.rand(batch_size, seq_len, hidden_size, device=device) < 0.85
    base_embeddings = base_embeddings * stability_mask.float() * 0.4
    base_embeddings = base_embeddings * 0.25
    volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.003 + 0.001
    risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.002
    return base_embeddings, volatility, risk_factors


def _create_commodities_data(batch_size, seq_len, hidden_size, device):
    """Generates sample data for Commodities."""
    base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.5
    t = torch.linspace(0, 8 * 3.14159, seq_len, device=device)
    cyclical_1 = torch.sin(t).unsqueeze(0).unsqueeze(-1) * 1.2
    cyclical_2 = torch.cos(t * 1.4).unsqueeze(0).unsqueeze(-1) * 0.9
    cyclical_3 = torch.sin(t * 0.6).unsqueeze(0).unsqueeze(-1) * 0.7
    base_embeddings[:, :, :5] += cyclical_1.expand(-1, -1, 5)
    base_embeddings[:, :, 5:10] += cyclical_2.expand(-1, -1, 5)
    base_embeddings[:, :, 10:15] += cyclical_3.expand(-1, -1, 5)
    shock_mask = torch.rand(batch_size, seq_len, 1, device=device) < 0.2
    shocks = shock_mask.float() * torch.randn(batch_size, seq_len, 1, device=device) * 1.5
    base_embeddings[:, :, :6] += shocks.expand(-1, -1, 6)
    base_embeddings = base_embeddings + torch.sin(base_embeddings) * 0.4
    volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.010 + 0.012
    risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.008
    return base_embeddings, volatility, risk_factors


def _create_fx_data(batch_size, seq_len, hidden_size, device):
    """Generates sample data for FX."""
    base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.15
    trend = torch.cumsum(torch.randn(batch_size, seq_len, 1, device=device) * 0.35, dim=1)
    base_embeddings[:, :, :14] += trend.expand(-1, -1, 14)
    direction = torch.randn(batch_size, 1, 1, device=device).sign() * 1.2
    base_embeddings[:, :, 8:] += direction.expand(-1, seq_len, hidden_size - 8)
    momentum = torch.zeros_like(trend)
    for i in range(1, seq_len):
        momentum[:, i, :] = momentum[:, i - 1, :] * 0.95 + torch.randn(batch_size, 1, device=device) * 0.4
    base_embeddings[:, :, :6] += momentum.expand(-1, -1, 6)
    base_embeddings = base_embeddings * 1.8
    base_embeddings = base_embeddings + base_embeddings.mean(dim=-1, keepdim=True) * 0.3
    volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.006 + 0.006
    risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.040
    return base_embeddings, volatility, risk_factors


def _create_derivatives_data(batch_size, seq_len, hidden_size, device):
    """Generates sample data for Derivatives."""
    base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 1.8
    base_embeddings = base_embeddings + torch.sin(base_embeddings * 3.0) * 1.0
    base_embeddings = base_embeddings + (base_embeddings ** 2).sign() * (base_embeddings.abs() ** 1.5) * 0.5
    convexity = torch.relu(base_embeddings) ** 2.0 - torch.relu(-base_embeddings) ** 2.0
    base_embeddings[:, :, :8] += convexity[:, :, :8] * 0.6
    vol_clustering = torch.zeros_like(base_embeddings[:, :, :8])
    for i in range(1, seq_len):
        vol_clustering[:, i, :] = vol_clustering[:, i - 1, :] * 0.9 + torch.randn(batch_size, 8, device=device) * 0.8
    base_embeddings[:, :, 8:] += vol_clustering
    volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.028 + 0.032
    risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.060
    return base_embeddings, volatility, risk_factors


def _create_credit_data(batch_size, seq_len, hidden_size, device):
    """Generates sample data for Credit."""
    base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.15
    stable_mask = torch.rand(batch_size, seq_len, hidden_size, device=device) < 0.70
    base_embeddings = base_embeddings * stable_mask.float() * 0.4
    jump_prob = 0.25
    jump_mask = torch.rand(batch_size, seq_len, 1, device=device) < jump_prob
    jumps = jump_mask.float() * torch.randn(batch_size, seq_len, 1, device=device) * 3.0
    base_embeddings[:, :, :10] += jumps.expand(-1, -1, 10)
    spread_widening = torch.zeros(batch_size, seq_len, 1, device=device)
    for i in range(1, seq_len):
        if torch.rand(1) < 0.30:
            spread_widening[:, i:, :] += torch.randn(1, device=device) * 1.5
    base_embeddings[:, :, 8:] += spread_widening.expand(-1, -1, hidden_size - 8)
    variance_pattern = torch.randn(batch_size, seq_len, 4, device=device) * 1.2
    base_embeddings[:, :, 12:16] += variance_pattern
    base_embeddings = base_embeddings - base_embeddings.mean(dim=-1, keepdim=True) * 0.7
    volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.004 + 0.003
    risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.010
    return base_embeddings, volatility, risk_factors


def _create_mixed_data(batch_size, seq_len, hidden_size, device):
    """Generates generic mixed sample data."""
    base_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.5
    volatility = torch.rand(batch_size, seq_len, 1, device=device) * 0.02 + 0.005
    risk_factors = torch.randn(batch_size, seq_len, 1, device=device) * 0.01
    return base_embeddings, volatility, risk_factors


def create_sample_data(batch_size=1, seq_len=30, hidden_size=16, asset_type='mixed', device='cuda'):
    """Creates highly distinct sample financial data with enhanced characteristics.

    Args:
        batch_size (int): The size of the batch.
        seq_len (int): The length of the sequence.
        hidden_size (int): The size of the hidden embeddings.
        asset_type (str): The type of asset data to generate. Supported values
            are 'equities', 'fixed_income', 'commodities', 'fx', 
            'derivatives', 'credit', and 'mixed'.
        device (str): The device to create tensors on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The generated sequence embeddings.
            - torch.Tensor: The generated market volatility.
            - torch.Tensor: The generated risk factors.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for data generation")

    device = 'cuda'  # Force GPU usage

    data_generators = {
        'equities': _create_equities_data,
        'fixed_income': _create_fixed_income_data,
        'commodities': _create_commodities_data,
        'fx': _create_fx_data,
        'derivatives': _create_derivatives_data,
        'credit': _create_credit_data,
        'mixed': _create_mixed_data,
    }

    generator_func = data_generators.get(asset_type.lower(), _create_mixed_data)

    return generator_func(batch_size, seq_len, hidden_size, device)


def analyze_routing_quality(model, asset_types, num_tests=5, seq_len=25):
    """Performs a comprehensive analysis of the model's routing quality.

    Args:
        model (FinanceMoEModel): The model to be analyzed.
        asset_types (list of str): A list of asset types to test.
        num_tests (int): The number of test runs for each asset type.
        seq_len (int): The sequence length for the generated data.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of average accuracy per asset type.
            - dict: A dictionary of average confidence per asset type.
            - dict: A dictionary with detailed results for each asset type.
    """
    print("FINANCE MOE ROUTING QUALITY ANALYSIS")
    print("=" * 60)

    overall_accuracy = {}
    overall_confidence = {}
    detailed_results = {}

    for asset_type in asset_types:
        print(f"\nTesting {asset_type.upper()}:")

        accuracies = []
        confidences = []

        for test_run in range(num_tests):
            # Create enhanced asset-specific sample data
            embeddings, volatility, risk = create_sample_data(
                batch_size=1, seq_len=seq_len, asset_type=asset_type, device=model.device
            )

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

            # Calculate accuracy and confidence
            correct_assignments = (assignments == expected_domain).sum()
            accuracy = correct_assignments / len(assignments) * 100
            accuracies.append(accuracy)

            confidence = np.max(probs, axis=1).mean()
            confidences.append(confidence)

            if test_run == 0:  # Show details for first run only
                domain_counts = np.bincount(assignments, minlength=6)
                print(f"  Run 1: {accuracy:.1f}% accuracy, {confidence:.3f} confidence")
                print(f"  Distribution: {domain_counts}")

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

        print(f"  Summary: {avg_accuracy:.1f}% ± {std_accuracy:.1f}% accuracy, {avg_confidence:.3f} confidence")

        # Performance assessment
        if avg_accuracy >= 95:
            status = "EXCELLENT"
        elif avg_accuracy >= 85:
            status = "VERY GOOD"
        elif avg_accuracy >= 75:
            status = "GOOD"
        elif avg_accuracy >= 60:
            status = "MODERATE"
        else:
            status = "POOR"
        print(f"  Status: {status}")

    # Overall performance summary
    print(f"\nOVERALL RESULTS:")
    avg_overall_accuracy = np.mean(list(overall_accuracy.values()))
    avg_overall_confidence = np.mean(list(overall_confidence.values()))

    print(f"Average Accuracy: {avg_overall_accuracy:.1f}%")
    print(f"Average Confidence: {avg_overall_confidence:.3f}")

    # Performance ranking
    sorted_assets = sorted(overall_accuracy.items(), key=lambda x: x[1], reverse=True)
    print(f"\nRanking:")
    for rank, (asset, accuracy) in enumerate(sorted_assets, 1):
        print(f"  {rank}. {asset.capitalize()}: {accuracy:.1f}%")

    return overall_accuracy, overall_confidence, detailed_results


# Test the model
if __name__ == "__main__":
    print("Finance MoE Model - GPU Performance Analysis")
    print("=" * 60)

    # Create model (will raise error if no GPU)
    model = FinanceMoEModel()
    print(f"Model initialized on GPU: {torch.cuda.get_device_name()}")

    # Test with different asset types
    asset_types = ['equities', 'fixed_income', 'commodities', 'fx', 'derivatives', 'credit']

    # Run comprehensive analysis
    accuracy_results, confidence_results, detailed_results = analyze_routing_quality(
        model, asset_types, num_tests=5, seq_len=30
    )

    # Performance summary
    avg_accuracy = np.mean(list(accuracy_results.values()))
    print(f"\nPerformance Summary:")
    print(f"Overall Accuracy: {avg_accuracy:.1f}% (Target: 85%)")
    print(f"\nGPU Memory used: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
