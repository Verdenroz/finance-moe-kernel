import compiler
from algorithm import vectorize, parallelize
from math import sqrt, exp, log, tanh
from memory import memset_zero, memcpy
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    ManagedTensorSlice,
    OutputTensor,
    foreach,
)
from utils.index import IndexList


# ================== HELPER FUNCTIONS ==================
@always_inline
fn softmax_inplace(data: OutputTensor, batch_idx: Int, seq_idx: Int, num_classes: Int):
    """Apply softmax to a row of probabilities."""
    var max_val = SIMD[DType.float16, 1](-1e9)

    # Find max for numerical stability
    for i in range(num_classes):
        var idx = IndexList[3](batch_idx, seq_idx, i)
        var val = data.load[1](idx)[0].cast[DType.float16]()
        if val > max_val:
            max_val = val

    # Compute exp and sum
    var sum_exp = SIMD[DType.float16, 1](0.0)
    for i in range(num_classes):
        var idx = IndexList[3](batch_idx, seq_idx, i)
        var val = data.load[1](idx)[0].cast[DType.float16]()
        var exp_val = exp(val - max_val)
        data.store[1](idx, SIMD[data.dtype, 1](exp_val.cast[data.dtype]()))
        sum_exp += exp_val

    # Normalize
    for i in range(num_classes):
        var idx = IndexList[3](batch_idx, seq_idx, i)
        var val = data.load[1](idx)[0].cast[DType.float16]()
        var normalized = val / sum_exp
        data.store[1](idx, SIMD[data.dtype, 1](normalized.cast[data.dtype]()))


@always_inline
fn compute_volatility(
    prices: InputTensor[dtype = DType.float16, rank=3],
    batch_idx: Int,
    seq_idx: Int,
    lookback: Int,
) -> SIMD[DType.float16, 1]:
    """Compute rolling volatility from price sequence."""
    var sum_returns = SIMD[DType.float16, 1](0.0)
    var sum_squared_returns = SIMD[DType.float16, 1](0.0)
    var count = SIMD[DType.float16, 1](0.0)

    for i in range(
        0 if seq_idx - lookback < 0 else seq_idx - lookback, seq_idx
    ):
        if i > 0:
            var curr_idx = IndexList[3](batch_idx, i, 0)
            var prev_idx = IndexList[3](batch_idx, i - 1, 0)
            var curr_price = prices.load[1](curr_idx)[0]
            var prev_price = prices.load[1](prev_idx)[0]

            if prev_price > SIMD[DType.float16, 1](0.0):
                var return_val = log(curr_price / prev_price)
                sum_returns += return_val
                sum_squared_returns += return_val * return_val
                count += SIMD[DType.float16, 1](1.0)

    if count > SIMD[DType.float16, 1](1.0):
        var mean_return = sum_returns / count
        var variance = (sum_squared_returns / count) - (
            mean_return * mean_return
        )
        var variance_val = variance if variance > SIMD[DType.float16, 1](
            1e-8
        ) else SIMD[DType.float16, 1](1e-8)
        return sqrt(variance_val)
    else:
        return SIMD[DType.float16, 1](0.1)  # Default volatility


# ================== FINANCE MACRO ROUTER ==================
@compiler.register("finance_macro_router")
struct FinanceMacroRouterOp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        domain_assignments: OutputTensor[dtype = DType.int32, rank=2],
        routing_probs: OutputTensor[dtype = DType.float16, rank=3],
        sequence_embeddings: InputTensor[dtype = DType.float16, rank=3],
        domain_router_weights: InputTensor[dtype = DType.float16, rank=2],
        domain_router_bias: InputTensor[dtype = DType.float16, rank=1],
        market_volatility: InputTensor[dtype = DType.float16, rank=2],
        risk_factors: InputTensor[dtype = DType.float16, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        var batch_size = domain_assignments.shape()[0]
        var seq_len = domain_assignments.shape()[1]
        var num_domains = routing_probs.shape()[2]
        var embedding_dim = sequence_embeddings.shape()[2]
        var num_risk_factors = risk_factors.shape()[2]

        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Compute domain routing logits
                var logits = List[SIMD[DType.float16, 1]]()

                for d in range(num_domains):
                    var logit = domain_router_bias.load[1](IndexList[1](d))[0]

                    # Add embedding contribution
                    for emb_dim in range(embedding_dim):
                        var weight_idx = IndexList[2](d, emb_dim)
                        var emb_idx = IndexList[3](batch_idx, seq_idx, emb_dim)
                        var weight = domain_router_weights.load[1](weight_idx)[
                            0
                        ]
                        var embedding = sequence_embeddings.load[1](emb_idx)[0]
                        logit += weight * embedding

                    # Add market volatility factor
                    var vol_idx = IndexList[2](batch_idx, seq_idx)
                    var volatility = market_volatility.load[1](vol_idx)[0]
                    var vol_factor = SIMD[DType.float16, 1](0.5) if d == 0 else SIMD[
                        DType.float16, 1
                    ](-0.3)
                    logit += volatility * vol_factor

                    # Add risk factor contributions
                    var max_risk_factors = (
                        num_risk_factors if num_risk_factors < 3 else 3
                    )
                    for rf in range(max_risk_factors):
                        var risk_idx = IndexList[3](batch_idx, seq_idx, rf)
                        var risk_val = risk_factors.load[1](risk_idx)[0]
                        var risk_factor = SIMD[DType.float16, 1](
                            0.2
                        ) if d == rf else SIMD[DType.float16, 1](-0.1)
                        logit += risk_val * risk_factor

                    logits.append(logit)

                # Store logits first
                for d in range(num_domains):
                    var prob_idx = IndexList[3](batch_idx, seq_idx, d)
                    routing_probs.store[1](
                        prob_idx,
                        SIMD[routing_probs.dtype, 1](
                            logits[d].cast[routing_probs.dtype]()
                        ),
                    )

                # Apply softmax to get probabilities
                softmax_inplace(
                    routing_probs, batch_idx, seq_idx, num_domains
                )

                # Assign to domain with highest probability
                var best_domain = 0
                var best_prob = routing_probs.load[1](
                    IndexList[3](batch_idx, seq_idx, 0)
                )[0].cast[DType.float16]()

                for d in range(1, num_domains):
                    var prob_idx = IndexList[3](batch_idx, seq_idx, d)
                    var prob = routing_probs.load[1](prob_idx)[0].cast[DType.float16]()
                    if prob > best_prob:
                        best_prob = prob
                        best_domain = d

                var domain_idx = IndexList[2](batch_idx, seq_idx)
                domain_assignments.store[1](
                    domain_idx, SIMD[domain_assignments.dtype, 1](best_domain)
                )


# ================== FINANCE MOE FFN ==================
@compiler.register("finance_moe_ffn")
struct FinanceMoEFFNOp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype = DType.float16, rank=3],
        expert_utilization: OutputTensor[dtype = DType.float16, rank=1],
        hidden_states: InputTensor[dtype = DType.float16, rank=3],
        gate_weights: InputTensor[dtype = DType.float16, rank=2],
        gate_bias: InputTensor[dtype = DType.float16, rank=1],
        up_weights: InputTensor[dtype = DType.float16, rank=3],
        down_weights: InputTensor[dtype = DType.float16, rank=3],
        domain_assignments: InputTensor[dtype = DType.int32, rank=2],
        market_regime: InputTensor[dtype = DType.float16, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var batch_size = output.shape()[0]
        var seq_len = output.shape()[1]
        var hidden_size = output.shape()[2]
        var num_experts = up_weights.shape()[0]
        var intermediate_size = up_weights.shape()[2]

        # Initialize expert utilization counters
        var expert_counts = List[SIMD[DType.float16, 1]]()
        for _ in range(num_experts):
            expert_counts.append(SIMD[DType.float16, 1](0.0))

        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Get domain assignment for this sequence position
                var domain_idx = IndexList[2](batch_idx, seq_idx)
                var assigned_domain = Int(
                    domain_assignments.load[1](domain_idx)[0]
                )

                # Compute gating probabilities
                var gate_logits = List[SIMD[DType.float16, 1]]()
                for expert_idx in range(num_experts):
                    var logit = gate_bias.load[1](IndexList[1](expert_idx))[0]

                    for hidden_idx in range(hidden_size):
                        var weight_idx = IndexList[2](expert_idx, hidden_idx)
                        var hidden_idx_3d = IndexList[3](
                            batch_idx, seq_idx, hidden_idx
                        )
                        var weight = gate_weights.load[1](weight_idx)[0]
                        var hidden_val = hidden_states.load[1](hidden_idx_3d)[0]
                        logit += weight * hidden_val

                    # Market regime adjustment
                    var regime_factor = market_regime.load[1](IndexList[1](0))[
                        0
                    ]
                    if expert_idx == assigned_domain:
                        logit += regime_factor * SIMD[DType.float16, 1](0.5)

                    gate_logits.append(logit)

                # Apply softmax to gate logits
                var max_logit = gate_logits[0]
                for i in range(1, num_experts):
                    if gate_logits[i] > max_logit:
                        max_logit = gate_logits[i]

                var sum_exp = SIMD[DType.float16, 1](0.0)
                var gate_probs = List[SIMD[DType.float16, 1]]()
                for i in range(num_experts):
                    var prob = exp(gate_logits[i] - max_logit)
                    gate_probs.append(prob)
                    sum_exp += prob

                for i in range(num_experts):
                    gate_probs[i] = gate_probs[i] / sum_exp

                # Compute expert outputs and mix them
                for hidden_idx in range(hidden_size):
                    var mixed_output = SIMD[DType.float16, 1](0.0)

                    for expert_idx in range(num_experts):
                        # Up projection (hidden -> intermediate)
                        var intermediate_sum = SIMD[DType.float16, 1](0.0)
                        for inter_idx in range(intermediate_size):
                            var up_weight_idx = IndexList[3](
                                expert_idx, hidden_idx, inter_idx
                            )
                            var hidden_idx_3d = IndexList[3](
                                batch_idx, seq_idx, hidden_idx
                            )
                            var up_weight = up_weights.load[1](up_weight_idx)[0]
                            var hidden_val = hidden_states.load[1](
                                hidden_idx_3d
                            )[0]
                            intermediate_sum += up_weight * hidden_val

                        # Apply activation (SiLU)
                        var activated = intermediate_sum * tanh(
                            intermediate_sum
                        )

                        # Down projection (intermediate -> hidden)
                        var expert_output = SIMD[DType.float16, 1](0.0)
                        for inter_idx in range(intermediate_size):
                            var down_weight_idx = IndexList[3](
                                expert_idx, inter_idx, hidden_idx
                            )
                            var down_weight = down_weights.load[1](
                                down_weight_idx
                            )[0]
                            expert_output += down_weight * activated

                        # Mix with gate probability
                        mixed_output += gate_probs[expert_idx] * expert_output

                        # Update expert utilization
                        expert_counts[expert_idx] += gate_probs[expert_idx]

                    # Store final output
                    var output_idx = IndexList[3](
                        batch_idx, seq_idx, hidden_idx
                    )
                    output.store[1](
                        output_idx,
                        SIMD[output.dtype, 1](
                            mixed_output.cast[output.dtype]()
                        ),
                    )

        # Normalize and store expert utilization
        var total_usage = SIMD[DType.float16, 1](0.0)
        for i in range(num_experts):
            total_usage += expert_counts[i]

        var safe_total = total_usage if total_usage > SIMD[DType.float16, 1](
            1e-8
        ) else SIMD[DType.float16, 1](1e-8)
        for i in range(num_experts):
            var util_idx = IndexList[1](i)
            var normalized_util = expert_counts[i] / safe_total
            expert_utilization.store[1](
                util_idx,
                SIMD[expert_utilization.dtype, 1](
                    normalized_util.cast[expert_utilization.dtype]()
                ),
            )


# ================== FINANCIAL LOAD BALANCING ==================
@compiler.register("financial_load_balancing")
struct FinancialLoadBalancingOp:
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
    ](
        aux_loss: OutputTensor,
        risk_metrics: OutputTensor,
        gate_probs: InputTensor[dtype=dtype, rank=3],
        expert_assignments: InputTensor[dtype=dtype, rank=3],
        expert_utilization: InputTensor[dtype=dtype, rank=1],
        risk_penalty_weight: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var batch_size = gate_probs.shape()[0]
        var seq_len = gate_probs.shape()[1]
        var num_experts = gate_probs.shape()[2]

        # Compute load balancing loss
        var mean_gate_prob = SIMD[dtype, 1](0.0)
        var mean_expert_frac = SIMD[dtype, 1](0.0)

        # Calculate mean gate probabilities
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                for expert_idx in range(num_experts):
                    var prob_idx = IndexList[3](batch_idx, seq_idx, expert_idx)
                    var prob = gate_probs.load[1](prob_idx)[0]
                    mean_gate_prob += prob
        var total_elements = SIMD[dtype, 1](batch_size * seq_len * num_experts)
        _ = mean_gate_prob / total_elements

        # Calculate mean expert assignment fractions
        for expert_idx in range(num_experts):
            var util_idx = IndexList[1](expert_idx)
            var util = expert_utilization.load[1](util_idx)[0]
            mean_expert_frac += util
        mean_expert_frac = mean_expert_frac / SIMD[dtype, 1](num_experts)

        # Load balancing loss (encourages uniform expert usage)
        var load_balance_loss = SIMD[dtype, 1](0.0)
        for expert_idx in range(num_experts):
            var util_idx = IndexList[1](expert_idx)
            var util = expert_utilization.load[1](util_idx)[0]
            var deviation = util - mean_expert_frac
            load_balance_loss += deviation * deviation

        # Risk concentration penalty
        var risk_penalty = risk_penalty_weight.load[1](IndexList[1](0))[0]
        var _ = SIMD[dtype, 1](0.0)  # concentration_penalty placeholder

        # Compute Herfindahl index for expert concentration
        var herfindahl = SIMD[dtype, 1](0.0)
        for expert_idx in range(num_experts):
            var util_idx = IndexList[1](expert_idx)
            var util = expert_utilization.load[1](util_idx)[0]
            herfindahl += util * util

        # Penalty increases with concentration (higher Herfindahl)
        var concentration_penalty = risk_penalty * herfindahl

        # Total auxiliary loss
        var total_aux_loss = load_balance_loss + concentration_penalty
        aux_loss.store[1](
            IndexList[1](0),
            SIMD[aux_loss.dtype, 1](total_aux_loss.cast[aux_loss.dtype]()),
        )

        # Compute risk metrics for each expert
        for expert_idx in range(num_experts):
            var risk_idx = IndexList[1](expert_idx)
            var util_idx = IndexList[1](expert_idx)
            var utilization = expert_utilization.load[1](util_idx)[0]

            # Risk metric: combination of over-utilization and under-utilization penalties
            var ideal_utilization = SIMD[dtype, 1](1.0) / SIMD[dtype, 1](
                num_experts
            )
            var util_diff = utilization - ideal_utilization
            var utilization_deviation = (
                util_diff if util_diff >= SIMD[dtype, 1](0.0) else -util_diff
            )

            # Higher risk for both over-utilized and under-utilized experts
            var risk_metric = utilization_deviation * SIMD[dtype, 1](2.0)

            # Add volatility-based risk component
            var over_util_threshold = ideal_utilization * SIMD[dtype, 1](1.5)
            var under_util_threshold = ideal_utilization * SIMD[dtype, 1](0.5)

            if utilization > over_util_threshold:
                risk_metric += SIMD[dtype, 1](
                    0.3
                )  # Penalty for over-concentration
            elif utilization < under_util_threshold:
                risk_metric += SIMD[dtype, 1](
                    0.2
                )  # Penalty for under-utilization

            risk_metrics.store[1](
                risk_idx,
                SIMD[risk_metrics.dtype, 1](
                    risk_metric.cast[risk_metrics.dtype]()
                ),
            )


# ================== MARKET REGIME DETECTION ==================
@compiler.register("market_regime_detector")
struct MarketRegimeDetectorOp:
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
    ](
        regime_indicators: OutputTensor,
        volatility_indicators: OutputTensor,
        price_sequences: InputTensor[dtype=dtype, rank=3],
        lookback_window: InputTensor[dtype = DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var batch_size = regime_indicators.shape()[0]
        var _ = regime_indicators.shape()[1]  # Bull, Bear, Sideways (num_regimes)
        var lookback = Int(lookback_window.load[1](IndexList[1](0))[0])

        for batch_idx in range(batch_size):
            # Compute price-based indicators
            var returns = List[SIMD[dtype, 1]]()
            var seq_len = price_sequences.shape()[1]

            # Calculate returns
            var max_lookback = (
                seq_len if seq_len < lookback + 1 else lookback + 1
            )
            for i in range(1, max_lookback):
                var curr_idx = IndexList[3](batch_idx, seq_len - i, 0)
                var prev_idx = IndexList[3](batch_idx, seq_len - i - 1, 0)
                var curr_price = price_sequences.load[1](curr_idx)[0]
                var prev_price = price_sequences.load[1](prev_idx)[0]

                if prev_price > SIMD[dtype, 1](0.0):
                    var return_val = (curr_price - prev_price) / prev_price
                    returns.append(return_val)

            if len(returns) == 0:
                # Default regime probabilities if no data
                var default_prob1 = SIMD[regime_indicators.dtype, 1](0.33)
                var default_prob2 = SIMD[regime_indicators.dtype, 1](0.33)
                var default_prob3 = SIMD[regime_indicators.dtype, 1](0.34)
                var default_vol = SIMD[volatility_indicators.dtype, 1](0.2)

                regime_indicators.store[1](
                    IndexList[2](batch_idx, 0), default_prob1
                )
                regime_indicators.store[1](
                    IndexList[2](batch_idx, 1), default_prob2
                )
                regime_indicators.store[1](
                    IndexList[2](batch_idx, 2), default_prob3
                )
                volatility_indicators.store[1](
                    IndexList[2](batch_idx, 0), default_vol
                )
                continue

            # Calculate statistics
            var mean_return = SIMD[dtype, 1](0.0)
            var sum_squared_returns = SIMD[dtype, 1](0.0)
            var positive_returns = 0
            var negative_returns = 0

            for i in range(len(returns)):
                var ret = returns[i]
                mean_return += ret
                sum_squared_returns += ret * ret
                if ret > SIMD[dtype, 1](0.0):
                    positive_returns += 1
                elif ret < SIMD[dtype, 1](0.0):
                    negative_returns += 1

            mean_return = mean_return / SIMD[dtype, 1](len(returns))
            var variance = (
                sum_squared_returns / SIMD[dtype, 1](len(returns))
            ) - (mean_return * mean_return)
            var safe_variance = variance if variance > SIMD[dtype, 1](
                1e-8
            ) else SIMD[dtype, 1](1e-8)
            var volatility = sqrt(safe_variance)

            # Regime classification logic
            var bull_prob = SIMD[dtype, 1](0.0)
            var bear_prob = SIMD[dtype, 1](0.0)
            var sideways_prob = SIMD[dtype, 1](0.0)

            # Bull market indicators
            if mean_return > SIMD[dtype, 1](0.01):  # Positive trend
                bull_prob += SIMD[dtype, 1](0.4)
            if positive_returns > negative_returns:  # More up days
                bull_prob += SIMD[dtype, 1](0.3)
            if volatility < SIMD[dtype, 1](0.2):  # Low volatility
                bull_prob += SIMD[dtype, 1](0.2)

            # Bear market indicators
            if mean_return < SIMD[dtype, 1](-0.01):  # Negative trend
                bear_prob += SIMD[dtype, 1](0.4)
            if negative_returns > positive_returns:  # More down days
                bear_prob += SIMD[dtype, 1](0.3)
            if volatility > SIMD[dtype, 1](0.3):  # High volatility
                bear_prob += SIMD[dtype, 1](0.2)

            # Sideways market (low trend, mixed signals)
            var abs_mean_return = (
                mean_return if mean_return
                >= SIMD[dtype, 1](0.0) else -mean_return
            )
            if abs_mean_return < SIMD[dtype, 1](0.005):  # Low trend
                sideways_prob += SIMD[dtype, 1](0.4)
            var return_diff = positive_returns - negative_returns
            var abs_return_diff = (
                return_diff if return_diff >= 0 else -return_diff
            )
            var threshold = Int(Float64(len(returns)) * 0.2)
            if abs_return_diff < threshold:  # Mixed signals
                sideways_prob += SIMD[dtype, 1](0.3)
            if volatility > SIMD[dtype, 1](0.15) and volatility < SIMD[
                dtype, 1
            ](
                0.25
            ):  # Medium volatility
                sideways_prob += SIMD[dtype, 1](0.2)

            # Normalize probabilities
            var total_prob = bull_prob + bear_prob + sideways_prob
            if total_prob > SIMD[dtype, 1](0.0):
                bull_prob = bull_prob / total_prob
                bear_prob = bear_prob / total_prob
                sideways_prob = sideways_prob / total_prob
            else:
                # Default equal probabilities
                bull_prob = SIMD[dtype, 1](0.33)
                bear_prob = SIMD[dtype, 1](0.33)
                sideways_prob = SIMD[dtype, 1](0.34)

            # Store regime probabilities
            regime_indicators.store[1](
                IndexList[2](batch_idx, 0),
                SIMD[regime_indicators.dtype, 1](
                    bull_prob.cast[regime_indicators.dtype]()
                ),
            )
            regime_indicators.store[1](
                IndexList[2](batch_idx, 1),
                SIMD[regime_indicators.dtype, 1](
                    bear_prob.cast[regime_indicators.dtype]()
                ),
            )
            regime_indicators.store[1](
                IndexList[2](batch_idx, 2),
                SIMD[regime_indicators.dtype, 1](
                    sideways_prob.cast[regime_indicators.dtype]()
                ),
            )

            # Store volatility
            volatility_indicators.store[1](
                IndexList[2](batch_idx, 0),
                SIMD[volatility_indicators.dtype, 1](
                    volatility.cast[volatility_indicators.dtype]()
                ),
            )
