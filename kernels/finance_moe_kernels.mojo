from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from math import exp, sqrt
from builtin.math import abs, max, min
from memory import stack_allocation
from algorithm import vectorize

# Model configuration constants
alias HIDDEN_SIZE = 16
alias NUM_DOMAINS = 6
alias VECTORIZATION_WIDTH = 4
alias MAX_LOGIT_VALUE = 20.0
alias MIN_LOGIT_VALUE = -20.0
alias NUMERICAL_EPSILON = 1e-8

@always_inline
fn compute_domain_score(
    domain: Int, 
    volatility: Float32, 
    risk: Float32,
    emb_mean: Float32, 
    emb_var: Float32, 
    emb_abs_mean: Float32
) -> Float32:
    """
    Compute domain-specific scoring for financial asset classification.
    
    Each domain has carefully tuned scoring logic based on real financial characteristics:
    - Volatility ranges specific to each asset class
    - Statistical patterns (variance, mean, absolute mean) that distinguish domains
    - Risk factor correlations specific to each financial instrument type
    
    Args:
        domain: Domain index (0-5)
        volatility: Market volatility measurement
        risk: Risk factor score
        emb_mean: Mean of embedding features
        emb_var: Variance of embedding features  
        emb_abs_mean: Absolute mean of embedding features
        
    Returns:
        Domain-specific score (higher = better match)
    """
    var score: Float32 = 0.0
    
    if domain == 0:  # EQUITIES - Moderate volatility with momentum patterns
        # Optimal volatility range: 1.5-4%
        if volatility >= 0.015 and volatility <= 0.04:
            score += 40.0  # Strong preference for equity volatility range
        else:
            score -= abs(volatility - 0.025) * 200.0  # Heavy penalty outside range
        
        # Momentum patterns (key equity characteristic)
        score += abs(emb_mean) * 25.0  # Reward directional momentum
        score += min(emb_var * 10.0, Float32(20.0))  # Moderate variance bonus, capped
        
    elif domain == 1:  # FIXED INCOME - Ultra-low volatility with high stability
        # Ultra-low volatility requirement: <0.5%
        if volatility < 0.005:
            score += 50.0 - volatility * 5000.0  # Strong bonus scaled by low volatility
        else:
            score -= volatility * 2000.0  # Heavy penalty for high volatility
        
        # Stability characteristics (key fixed income traits)
        score += max(Float32(0.0), (Float32(0.2) - emb_var)) * 50.0  # Reward low variance
        score += max(Float32(0.0), (Float32(0.01) - abs(risk))) * 100.0  # Reward low risk
        score += max(Float32(0.0), (Float32(0.1) - emb_abs_mean)) * 30.0  # Reward stability
        
    elif domain == 2:  # COMMODITIES - Moderate volatility with strong cyclical patterns
        # Optimal volatility range: 1-2.5%
        if volatility >= 0.01 and volatility <= 0.025:
            score += 35.0  # Preference for commodity volatility range
        else:
            score -= abs(volatility - 0.0175) * 150.0  # Penalty outside range
        
        # Cyclical patterns (key commodity characteristic)
        if emb_var > 0.5:
            score += min((emb_var - Float32(0.5)) * 20.0, Float32(40.0))  # Reward high variance
        else:
            score -= (Float32(0.5) - emb_var) * 25.0  # Penalty for low variance
        
        score += emb_abs_mean * 15.0  # Moderate absolute mean bonus
        
    elif domain == 3:  # FX - Low-moderate volatility with strong trending behavior
        # Optimal volatility range: 0.5-1.3%
        if volatility >= 0.005 and volatility <= 0.013:
            score += 40.0  # Strong preference for FX volatility range
        else:
            score -= abs(volatility - 0.009) * 200.0  # Heavy penalty outside range
        
        # Strong trending behavior (key FX differentiator)
        if abs(emb_mean) > 1.5:  # Very strong trend
            score += 60.0 + (abs(emb_mean) - 1.5) * 20.0
        elif abs(emb_mean) > 0.8:  # Strong trend
            score += 40.0 + (abs(emb_mean) - 0.8) * 25.0
        elif abs(emb_mean) > 0.3:  # Moderate trend
            score += 15.0 + (abs(emb_mean) - 0.3) * 30.0
        else:  # Weak trend - strong penalty
            score -= (0.3 - abs(emb_mean)) * 50.0
        
        # Risk factor considerations
        score += max(Float32(0.0), (Float32(0.05) - abs(risk))) * 15.0
        
        # Distinguish from commodities (penalize very high variance)
        if emb_var > 3.0:
            score -= (emb_var - 3.0) * 10.0
        
    elif domain == 4:  # DERIVATIVES - High volatility with complex patterns
        # High volatility requirement: >3%
        if volatility > 0.03:
            score += min(volatility * 150.0, Float32(80.0))  # Strong bonus, capped
        else:
            score -= (Float32(0.03) - volatility) * 300.0  # Heavy penalty for low volatility
        
        # Complex patterns (key derivative characteristics)
        score += min(emb_var * 20.0, Float32(60.0))  # High variance bonus
        score += min(abs(risk) * 30.0, Float32(50.0))  # High risk bonus
        score += min(emb_abs_mean * 8.0, Float32(25.0))  # Complexity bonus
        
    else:  # CREDIT - Low volatility with jump patterns and moderate variance
        # Optimal volatility range: 0.2-0.8% (higher than fixed income)
        if volatility >= 0.002 and volatility <= 0.008:
            score += 40.0  # Strong preference for credit volatility range
        else:
            score -= abs(volatility - 0.005) * 250.0  # Heavy penalty outside range
        
        # Moderate variance requirement (jump patterns)
        if emb_var >= 0.3 and emb_var <= 3.0:
            score += 35.0 - abs(emb_var - 1.2) * 8.0  # Optimal around 1.2
        else:
            score -= abs(emb_var - 1.2) * 20.0  # Penalty for extreme variance
        
        # Distinguish from FX (penalize strong trending)
        if abs(emb_mean) > 0.8:
            score -= abs(emb_mean) * 25.0  # Penalty for strong trends
        else:
            score += (Float32(0.8) - abs(emb_mean)) * 5.0  # Small bonus for low trending
        
        # Credit-specific risk characteristics
        score += max(Float32(0.0), (Float32(0.02) - abs(abs(risk) - Float32(0.01)))) * 50.0
    
    return score

@always_inline 
fn compute_heuristic_domain(
    volatility: Float32, 
    emb_var: Float32, 
    emb_mean: Float32
) -> Int32:
    """
    Intelligent heuristic fallback for low-confidence routing decisions.
    
    When the main scoring system produces low confidence predictions, this function
    provides rule-based classification using clear decision boundaries based on
    the most distinguishing characteristics of each financial domain.
    
    Decision tree logic:
    1. High volatility → Derivatives (clear separator)
    2. Ultra-low volatility → Fixed Income vs Credit (variance-based)
    3. Low-moderate volatility → FX vs Credit (trend-based)
    4. Moderate volatility → Commodities vs Equities (variance-based)
    
    Args:
        volatility: Market volatility measurement
        emb_var: Embedding variance (cyclical/complexity indicator)
        emb_mean: Embedding mean (trend strength indicator)
        
    Returns:
        Domain ID (0-5) based on heuristic rules
    """
    # Clear high volatility separator: Derivatives
    if volatility > 0.03:
        return Int32(4)  # Derivatives
    
    # Ultra-low volatility: Fixed Income vs Credit decision
    elif volatility < 0.005:
        if emb_var < 0.15:  # Very stable patterns
            return Int32(1)  # Fixed Income
        else:  # Some variance (jump patterns)
            return Int32(5)  # Credit
    
    # Low-moderate volatility range: FX vs Credit decision
    elif volatility >= 0.005 and volatility <= 0.013:
        # Strong trending behavior strongly indicates FX
        if abs(emb_mean) > 1.5:
            return Int32(3)  # FX - very strong trend
        elif abs(emb_mean) > 0.8 and emb_var < 5.0:
            return Int32(3)  # FX - strong trend with reasonable variance
        elif abs(emb_mean) < 0.6 and emb_var >= 0.5 and emb_var <= 4.0:
            return Int32(5)  # Credit - low trend with moderate variance
        else:
            return Int32(3)  # Default to FX for this volatility range
    
    # Moderate volatility range: Commodities vs Equities decision
    elif volatility >= 0.01 and volatility <= 0.025:
        if emb_var > 0.6:  # High variance indicates cyclical patterns
            return Int32(2)  # Commodities
        else:  # Lower variance indicates momentum patterns
            return Int32(0)  # Equities
    
    # Higher moderate volatility: likely Equities
    elif volatility > 0.015 and volatility <= 0.04:
        return Int32(0)  # Equities
    
    # Remaining high volatility: Derivatives
    elif volatility > 0.025:
        return Int32(4)  # Derivatives
    
    # Fallback to FX as most versatile domain
    else:
        return Int32(3)  # FX

@register("finance_router")
struct FinanceRouter:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        domain_assignments: OutputTensor[dtype = DType.int32, rank=2],
        routing_probs: OutputTensor[dtype = DType.float32, rank=3],
        sequence_embeddings: InputTensor[dtype = DType.float32, rank=3],
        domain_router_weights: InputTensor[dtype = DType.float32, rank=2],
        domain_router_bias: InputTensor[dtype = DType.float32, rank=1],
        market_volatility: InputTensor[dtype = DType.float32, rank=3],
        risk_factors: InputTensor[dtype = DType.float32, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn compute_routing[
            simd_width: Int
        ](idx: IndexList[domain_assignments.rank]) -> SIMD[DType.int32, simd_width]:
            var b = idx[0]
            var t = idx[1]
            
            # Use stack allocation for better memory management
            var domain_logits = stack_allocation[NUM_DOMAINS, Float32]()
            var domain_probs = stack_allocation[NUM_DOMAINS, Float32]()
            
            # Extract market conditions with bounds checking
            var volatility = market_volatility.load[1](IndexList[3](b, t, 0))[0]
            var risk = risk_factors.load[1](IndexList[3](b, t, 0))[0]
            
            # Clamp values for numerical stability
            volatility = max(min(volatility, Float32(0.1)), Float32(0.0001))
            risk = max(min(risk, Float32(0.2)), Float32(-0.2))
            
            # Compute embedding statistics efficiently
            var embedding_sum: Float32 = 0.0
            var embedding_sq_sum: Float32 = 0.0
            var embedding_abs_sum: Float32 = 0.0
            
            # Vectorized computation of embedding features
            @parameter
            fn compute_features[width: Int](i: Int):
                if i + width <= HIDDEN_SIZE:
                    var vals = sequence_embeddings.load[width](IndexList[3](b, t, i))
                    for j in range(width):
                        var val = vals[j]
                        embedding_sum += val
                        embedding_sq_sum += val * val
                        embedding_abs_sum += abs(val)
                else:
                    # Handle remaining elements
                    for j in range(i, HIDDEN_SIZE):
                        var val = sequence_embeddings.load[1](IndexList[3](b, t, j))[0]
                        embedding_sum += val
                        embedding_sq_sum += val * val
                        embedding_abs_sum += abs(val)
            
            vectorize[compute_features, 4](HIDDEN_SIZE)
            
            # Compute statistics with numerical stability
            var embedding_mean = embedding_sum / Float32(HIDDEN_SIZE)
            var embedding_variance = max((embedding_sq_sum / Float32(HIDDEN_SIZE)) - (embedding_mean * embedding_mean), Float32(0.001))
            var embedding_abs_mean = embedding_abs_sum / Float32(HIDDEN_SIZE)
            
            # Enhanced domain logit computation
            var best_domain: Int32 = 0
            var best_score: Float32 = -1e6
            
            # Compute logits for each domain
            for d in range(NUM_DOMAINS):
                # Base linear projection
                var logit: Float32 = domain_router_bias.load[1](IndexList[1](d))[0]
                
                # Vectorized dot product
                @parameter
                fn dot_product[width: Int](i: Int):
                    if i + width <= HIDDEN_SIZE:
                        var x_vals = sequence_embeddings.load[width](IndexList[3](b, t, i))
                        var w_vals = domain_router_weights.load[width](IndexList[2](i, d))
                        for j in range(width):
                            logit += x_vals[j] * w_vals[j]
                    else:
                        # Handle remaining elements
                        for j in range(i, HIDDEN_SIZE):
                            var x_val = sequence_embeddings.load[1](IndexList[3](b, t, j))[0]
                            var w_val = domain_router_weights.load[1](IndexList[2](j, d))[0]
                            logit += x_val * w_val
                
                vectorize[dot_product, 4](HIDDEN_SIZE)
                
                # Add enhanced domain-specific scoring
                var domain_score = compute_domain_score(
                    d, volatility, risk, 
                    embedding_mean, embedding_variance, embedding_abs_mean
                )
                logit += domain_score
                
                # Apply temperature scaling and numerical stability
                var temperature: Float32 = 1.0  # Reduced temperature for sharper decisions
                logit /= temperature
                logit = max(min(logit, Float32(20.0)), Float32(-20.0))  # Wider clipping range
                
                domain_logits[d] = logit
                
                if logit > best_score:
                    best_score = logit
                    best_domain = d
            
            # Compute softmax probabilities with improved numerical stability
            var total_exp: Float32 = 0.0
            for d in range(NUM_DOMAINS):
                var exp_logit = exp(domain_logits[d] - best_score)
                domain_probs[d] = exp_logit
                total_exp += exp_logit
            
            # Normalize and store probabilities
            var total_exp_safe = max(total_exp, Float32(1e-8))
            for d in range(NUM_DOMAINS):
                var prob = domain_probs[d] / total_exp_safe
                routing_probs.store[1](IndexList[3](b, t, d), prob)
            
            # Enhanced confidence-based fallback with improved threshold
            var max_prob = domain_probs[best_domain] / total_exp_safe
            if max_prob < Float32(0.35):  # Adjusted threshold
                best_domain = compute_heuristic_domain(volatility, embedding_variance, embedding_mean)
            
            return SIMD[DType.int32, simd_width](best_domain)

        foreach[
            compute_routing, 
            target=target, 
            simd_width=1  # Conservative SIMD width for better compatibility
        ](domain_assignments, ctx)