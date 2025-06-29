from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from math import exp, sqrt
from builtin.math import abs, max, min
from memory import stack_allocation
from algorithm import vectorize

alias HIDDEN_SIZE = 16
alias NUM_DOMAINS = 6

@always_inline
fn compute_domain_score(
    domain: Int, volatility: Float32, risk: Float32,
    emb_mean: Float32, emb_var: Float32, emb_abs_mean: Float32
) -> Float32:
    """
    Enhanced domain scoring with improved asset classification logic.
    Each domain has specific volatility ranges and feature patterns.
    """
    var score: Float32 = 0.0
    
    if domain == 0:  # Equities - moderate volatility (1.5-4%), momentum patterns
        # Strong volatility preference for equity range
        if volatility >= 0.015 and volatility <= 0.04:
            score += 40.0  # Strong bonus for optimal range
        else:
            score -= abs(volatility - 0.025) * 200.0  # Heavy penalty outside range
        
        # Moderate momentum and variance bonus
        score += abs(emb_mean) * 25.0  # Momentum component
        score += min(emb_var * 10.0, Float32(20.0))  # Moderate variance bonus, capped
        
    elif domain == 1:  # Fixed Income - very low volatility (<0.5%), high stability
        # Ultra-low volatility requirement
        if volatility < 0.005:
            score += 50.0 - volatility * 5000.0  # Strong bonus, scaled by how low
        else:
            score -= volatility * 2000.0  # Heavy penalty for high volatility
        
        # Stability bonuses - penalize high variance and risk
        score += max(Float32(0.0), (Float32(0.2) - emb_var)) * 50.0  # Bonus for low variance
        score += max(Float32(0.0), (Float32(0.01) - abs(risk))) * 100.0  # Bonus for low risk
        score += max(Float32(0.0), (Float32(0.1) - emb_abs_mean)) * 30.0  # Bonus for stability
        
    elif domain == 2:  # Commodities - moderate volatility (1-2.5%), cyclical patterns
        # Optimal volatility range
        if volatility >= 0.01 and volatility <= 0.025:
            score += 35.0
        else:
            score -= abs(volatility - 0.0175) * 150.0
        
        # Strong variance requirement (cyclical patterns)
        if emb_var > 0.5:
            score += min((emb_var - Float32(0.5)) * 20.0, Float32(40.0))  # Bonus for high variance
        else:
            score -= (Float32(0.5) - emb_var) * 25.0  # Penalty for low variance
        
        # Moderate absolute mean bonus
        score += emb_abs_mean * 15.0
        
    elif domain == 3:  # FX - low-moderate volatility (0.5-1.3%), strong trending
        # Optimal volatility range
        if volatility >= 0.005 and volatility <= 0.013:
            score += 40.0
        else:
            score -= abs(volatility - 0.009) * 200.0
        
        # Very strong trending requirement (key FX differentiator)
        if abs(emb_mean) > 1.5:  # Very strong trend
            score += 60.0 + (abs(emb_mean) - 1.5) * 20.0
        elif abs(emb_mean) > 0.8:  # Strong trend
            score += 40.0 + (abs(emb_mean) - 0.8) * 25.0
        elif abs(emb_mean) > 0.3:  # Moderate trend
            score += 15.0 + (abs(emb_mean) - 0.3) * 30.0
        else:  # Weak trend - strong penalty
            score -= (0.3 - abs(emb_mean)) * 50.0
        
        # Moderate risk tolerance
        score += max(Float32(0.0), (Float32(0.05) - abs(risk))) * 15.0
        
        # Penalize very high variance (distinguish from commodities)
        if emb_var > 3.0:
            score -= (emb_var - 3.0) * 10.0
        
    elif domain == 4:  # Derivatives - high volatility (>3%), complex patterns
        # High volatility requirement
        if volatility > 0.03:
            score += min(volatility * 150.0, Float32(80.0))  # Strong bonus, capped
        else:
            score -= (Float32(0.03) - volatility) * 300.0  # Heavy penalty
        
        # High variance and complexity bonuses
        score += min(emb_var * 20.0, Float32(60.0))  # High variance bonus
        score += min(abs(risk) * 30.0, Float32(50.0))  # High risk bonus
        
        # Bonus for high absolute mean (complex patterns)
        score += min(emb_abs_mean * 8.0, Float32(25.0))
        
    else:  # Credit - low volatility (0.2-0.8%), jump patterns, moderate variance
        # Optimal volatility range (higher than fixed income)
        if volatility >= 0.002 and volatility <= 0.008:
            score += 40.0
        else:
            score -= abs(volatility - 0.005) * 250.0
        
        # Moderate variance requirement (higher than fixed income, lower than commodities)
        if emb_var >= 0.3 and emb_var <= 3.0:
            score += 35.0 - abs(emb_var - 1.2) * 8.0  # Optimal around 1.2
        else:
            score -= abs(emb_var - 1.2) * 20.0
        
        # Penalize strong trending (distinguish from FX)
        if abs(emb_mean) > 0.8:
            score -= abs(emb_mean) * 25.0
        else:
            score += (Float32(0.8) - abs(emb_mean)) * 5.0  # Small bonus for low trending
        
        # Moderate risk bonus
        score += max(Float32(0.0), (Float32(0.02) - abs(abs(risk) - Float32(0.01)))) * 50.0
    
    return score

@always_inline 
fn compute_heuristic_domain(volatility: Float32, emb_var: Float32, emb_mean: Float32) -> Int32:
    """
    Improved heuristic fallback with better domain separation logic.
    Used when confidence is low to make more accurate predictions.
    """
    # High volatility -> Derivatives (clear separation)
    if volatility > 0.03:
        return Int32(4)
    
    # Very low volatility -> Fixed Income or Credit
    elif volatility < 0.005:
        if emb_var < 0.15:  # Very stable -> Fixed Income
            return Int32(1)
        else:  # Some variance -> Credit
            return Int32(5)
    
    # Low-moderate volatility range (0.005-0.013) -> FX vs Credit decision
    elif volatility >= 0.005 and volatility <= 0.013:
        # Strong trending strongly indicates FX
        if abs(emb_mean) > 1.5:
            return Int32(3)  # FX
        # Moderate trending with reasonable variance -> FX
        elif abs(emb_mean) > 0.8 and emb_var < 5.0:
            return Int32(3)  # FX
        # Low trending with moderate variance -> Credit
        elif abs(emb_mean) < 0.6 and emb_var >= 0.5 and emb_var <= 4.0:
            return Int32(5)  # Credit
        else:
            return Int32(3)  # Default to FX
    
    # Moderate volatility range (0.01-0.025) -> Commodities vs Equities
    elif volatility >= 0.01 and volatility <= 0.025:
        # High variance indicates cyclical patterns -> Commodities
        if emb_var > 0.6:
            return Int32(2)  # Commodities
        else:
            return Int32(0)  # Equities
    
    # Higher moderate volatility -> likely Equities unless very high
    elif volatility > 0.015 and volatility <= 0.04:
        return Int32(0)  # Equities
    
    # Very high volatility -> Derivatives
    elif volatility > 0.025:
        return Int32(4)  # Derivatives
    
    # Default fallback
    else:
        return Int32(3)  # FX as reasonable default

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