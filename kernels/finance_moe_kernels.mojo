from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from math import exp, sqrt, log
from builtin.math import abs, max, min
from memory import stack_allocation
from algorithm import vectorize

alias HIDDEN_SIZE = 16
alias NUM_DOMAINS = 6

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
        market_volatility: InputTensor[dtype = DType.float32, rank=2],
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

            # Use stack allocation for domain probabilities
            var domain_logits = stack_allocation[NUM_DOMAINS, Float32]()
            var domain_probs = stack_allocation[NUM_DOMAINS, Float32]()
            
            # Extract market conditions
            var volatility = market_volatility.load[1](IndexList[2](b, t))[0]
            var risk = risk_factors.load[1](IndexList[3](b, t, 0))[0]
            
            # Compute embedding statistics for asset class identification
            var embedding_sum: Float32 = 0.0
            var embedding_sq_sum: Float32 = 0.0
            var embedding_abs_sum: Float32 = 0.0
            
            # Vectorized computation of embedding features
            @parameter
            fn compute_features[width: Int](i: Int):
                var vals = sequence_embeddings.load[width](IndexList[3](b, t, i))
                for j in range(width):
                    if i + j < HIDDEN_SIZE:
                        var val = vals[j]
                        embedding_sum += val
                        embedding_sq_sum += val * val
                        embedding_abs_sum += abs(val)
            
            vectorize[compute_features, 4](HIDDEN_SIZE)
            
            var embedding_mean = embedding_sum / HIDDEN_SIZE
            var embedding_variance = (embedding_sq_sum / HIDDEN_SIZE) - (embedding_mean * embedding_mean)
            var embedding_abs_mean = embedding_abs_sum / HIDDEN_SIZE
            
            # Compute domain logits with enhanced asset-specific logic
            var best_domain: Int32 = 0
            var best_score: Float32 = -1e9
            var total_exp: Float32 = 0.0
            
            for d in range(NUM_DOMAINS):
                # Base linear projection (vectorized)
                var logit: Float32 = domain_router_bias.load[1](IndexList[1](d))[0]
                
                @parameter
                fn dot_product[width: Int](i: Int):
                    var x_vals = sequence_embeddings.load[width](IndexList[3](b, t, i))
                    var w_vals = domain_router_weights.load[width](IndexList[2](i, d))
                    for j in range(width):
                        if i + j < HIDDEN_SIZE:
                            logit += x_vals[j] * w_vals[j]
                
                vectorize[dot_product, 4](HIDDEN_SIZE)
                
                # Asset-specific adjustments with refined volatility ranges
                if d == 0:  # Equities - moderate volatility (1.5-4%), momentum patterns
                    # Target volatility range: 1.5% - 4%
                    var vol_in_range = (volatility >= 0.015 and volatility <= 0.04)
                    var vol_score: Float32 = 0.0
                    if vol_in_range:
                        vol_score = 25.0  # Strong bonus for being in range
                    else:
                        vol_score = -abs(volatility - 0.025) * 100.0  # Penalty for being outside
                    
                    var momentum_score = abs(embedding_mean) * 20.0
                    var diversity_bonus = embedding_variance * 15.0
                    logit += vol_score + momentum_score + diversity_bonus
                    
                elif d == 1:  # Fixed Income - very low volatility (<0.5%)
                    # Target volatility range: < 0.5%
                    var is_low_vol = volatility < 0.005
                    var vol_score: Float32 = 0.0
                    if is_low_vol:
                        vol_score = 30.0 - volatility * 2000.0  # Reward very low vol
                    else:
                        vol_score = -volatility * 1000.0  # Heavy penalty for high vol
                    
                    var stability_bonus = (0.1 - embedding_variance) * 25.0
                    var low_risk_bonus = (0.005 - abs(risk)) * 40.0
                    logit += vol_score + stability_bonus + low_risk_bonus
                    
                elif d == 2:  # Commodities - moderate volatility (1-2.5%), cyclical
                    # Target volatility range: 1% - 2.5%
                    var vol_in_range = (volatility >= 0.01 and volatility <= 0.025)
                    var vol_score: Float32 = 0.0
                    if vol_in_range:
                        vol_score = 20.0
                    else:
                        vol_score = -abs(volatility - 0.0175) * 80.0
                    
                    # Look for cyclical patterns in embeddings
                    var cyclical_bonus = abs(embedding_variance - 0.4) * 15.0
                    var trend_score = embedding_abs_mean * 18.0
                    logit += vol_score + cyclical_bonus + trend_score
                    
                elif d == 3:  # FX - low-moderate volatility (0.5-1.3%), trending
                    # Target volatility range: 0.5% - 1.3%
                    var vol_in_range = (volatility >= 0.005 and volatility <= 0.013)
                    var vol_score: Float32 = 0.0
                    if vol_in_range:
                        vol_score = 22.0  # Strong bonus for FX range
                    else:
                        vol_score = -abs(volatility - 0.009) * 120.0
                    
                    var trend_strength = abs(embedding_mean) * 25.0
                    var directional_bias = risk * 30.0
                    logit += vol_score + trend_strength + directional_bias
                    
                elif d == 4:  # Derivatives - high volatility (>3%), complex
                    # Target volatility range: > 3%
                    var is_high_vol = volatility > 0.03
                    var vol_score: Float32 = 0.0
                    if is_high_vol:
                        vol_score = min(volatility * 100.0, 30.0)  # Reward high vol, capped
                    else:
                        vol_score = (volatility - 0.03) * 200.0  # Penalty if below 3%
                    
                    var complexity_bonus = embedding_variance * 35.0
                    var risk_seeking = abs(risk) * 40.0
                    logit += vol_score + complexity_bonus + risk_seeking
                    
                else:  # Credit - low volatility (0.2-0.8%), jump patterns
                    # Target volatility range: 0.2% - 0.8%
                    var vol_in_range = (volatility >= 0.002 and volatility <= 0.008)
                    var vol_score: Float32 = 0.0
                    if vol_in_range:
                        vol_score = 18.0
                    else:
                        vol_score = -abs(volatility - 0.005) * 150.0
                    
                    # Look for jump patterns (high variance with low mean)
                    var jump_pattern = embedding_variance * 20.0 - abs(embedding_mean) * 10.0
                    var credit_risk_score = abs(risk) * 15.0
                    logit += vol_score + jump_pattern + credit_risk_score
                
                # Apply temperature and numerical stability
                var temperature: Float32 = 1.2  # Higher temperature for less aggressive decisions
                logit /= temperature
                logit = max(min(logit, 15.0), -15.0)  # Wider clipping range
                
                domain_logits[d] = logit
                
                if logit > best_score:
                    best_score = logit
                    best_domain = d
            
            # Compute softmax probabilities
            for d in range(NUM_DOMAINS):
                var exp_logit = exp(domain_logits[d] - best_score)  # Numerical stability
                domain_probs[d] = exp_logit
                total_exp += exp_logit
            
            # Normalize and store probabilities
            for d in range(NUM_DOMAINS):
                var prob = domain_probs[d] / max(total_exp, 1e-8)
                routing_probs.store[1](IndexList[3](b, t, d), prob)
            
            # Decision confidence check - use simple rules if uncertain
            var max_prob = domain_probs[best_domain] / max(total_exp, 1e-8)
            
            if max_prob < 0.4:  # Low confidence, use precise heuristic rules
                if volatility > 0.03:  # > 3% volatility
                    best_domain = 4  # Derivatives
                elif volatility < 0.005:  # < 0.5% volatility
                    if embedding_variance < 0.2:  # Low complexity
                        best_domain = 1  # Fixed Income
                    else:
                        best_domain = 5  # Credit (jump patterns)
                elif volatility >= 0.005 and volatility <= 0.013:  # 0.5-1.3% volatility
                    if abs(embedding_mean) > 0.3:  # Strong trend
                        best_domain = 3  # FX
                    else:
                        best_domain = 5  # Credit
                elif volatility >= 0.01 and volatility <= 0.025:  # 1-2.5% volatility
                    if embedding_variance > 0.4:  # High complexity/cyclical
                        best_domain = 2  # Commodities
                    else:
                        best_domain = 0  # Equities
                elif volatility > 0.015:  # 1.5%+ volatility
                    if volatility < 0.025:  # Under 2.5%
                        best_domain = 0  # Equities
                    else:
                        best_domain = 4  # Derivatives
                else:
                    best_domain = 3  # FX (default for moderate conditions)
            
            return SIMD[DType.int32, simd_width](best_domain)

        foreach[
            compute_routing, 
            target=target, 
            simd_width=1
        ](domain_assignments, ctx)