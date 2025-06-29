from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from math import exp, sqrt, tanh, log
from builtin.math import abs, max, min
from memory import stack_allocation
from algorithm import vectorize, parallelize
from sys import simdwidthof

# Enhanced model configuration
alias HIDDEN_SIZE = 32
alias NUM_DOMAINS = 6
alias VECTORIZATION_WIDTH = 8
alias MAX_LOGIT_VALUE = 15.0
alias MIN_LOGIT_VALUE = -15.0
alias NUMERICAL_EPSILON = 1e-7
alias SIMD_WIDTH = 8
alias TILE_SIZE = 32
alias PARALLEL_WORK_SIZE = 4


@register("mojo_feature_extractor")
struct MojoFeatureExtractor:
    """Mojo-powered feature extraction replacing PyTorch Sequential layers."""
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output_features: OutputTensor[dtype = DType.float32, rank=3],  # [batch, seq, hidden*2]
        input_embeddings: InputTensor[dtype = DType.float32, rank=3],   # [batch, seq, hidden]
        weights: InputTensor[dtype = DType.float32, rank=2],           # [hidden, hidden*2]
        bias: InputTensor[dtype = DType.float32, rank=1],              # [hidden*2]
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn mojo_extract_features[
            width: Int
        ](idx: IndexList[output_features.rank]) -> SIMD[DType.float32, width]:
            var b = idx[0]
            var t = idx[1]
            var d = idx[2]  # output dimension (0 to hidden*2-1)

            # High-performance matrix multiply with ReLU activation
            var accumulator = stack_allocation[SIMD_WIDTH, Float32]()
            var bias_val = bias.load[1](IndexList[1](d))[0]

            # Initialize accumulator with bias
            @parameter
            for i in range(SIMD_WIDTH):
                accumulator[i] = bias_val / Float32(SIMD_WIDTH)

            # Vectorized dot product computation
            @parameter
            fn optimized_matmul[v_width: Int](i: Int):
                if i + v_width <= HIDDEN_SIZE:
                    var x_vals = input_embeddings.load[v_width](IndexList[3](b, t, i))
                    var w_vals = weights.load[v_width](IndexList[2](i, d))

                    @parameter
                    for j in range(v_width):
                        var lane_idx = j % SIMD_WIDTH
                        accumulator[lane_idx] += x_vals[j] * w_vals[j]
                else:
                    # Handle remainder
                    for j in range(i, HIDDEN_SIZE):
                        var x_val = input_embeddings.load[1](IndexList[3](b, t, j))[0]
                        var w_val = weights.load[1](IndexList[2](j, d))[0]
                        accumulator[0] += x_val * w_val

            vectorize[optimized_matmul, SIMD_WIDTH](HIDDEN_SIZE)

            # Reduce accumulator
            var result: Float32 = 0.0
            @parameter
            for i in range(SIMD_WIDTH):
                result += accumulator[i]

            # Apply ReLU activation
            result = max(result, Float32(0.0))

            return SIMD[DType.float32, width](result)

        foreach[
            mojo_extract_features,
            target=target,
            simd_width=1
        ](output_features, ctx)


@register("mojo_all_routers")
struct MojoAllRouters:
    """Single Mojo kernel handling ALL router computations."""
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        base_logits: OutputTensor[dtype = DType.float32, rank=3],
        vol_logits: OutputTensor[dtype = DType.float32, rank=3],
        risk_logits: OutputTensor[dtype = DType.float32, rank=3],
        stats_logits: OutputTensor[dtype = DType.float32, rank=3],
        extracted_features: InputTensor[dtype = DType.float32, rank=3],    # [batch, seq, hidden*2]
        market_volatility: InputTensor[dtype = DType.float32, rank=3],     # [batch, seq, 1]
        risk_factors: InputTensor[dtype = DType.float32, rank=3],          # [batch, seq, 1]
        embedding_stats: InputTensor[dtype = DType.float32, rank=3],       # [batch, seq, 3]
        domain_classifier_weights: InputTensor[dtype = DType.float32, rank=2],  # [hidden*2, num_domains]
        domain_classifier_bias: InputTensor[dtype = DType.float32, rank=1],     # [num_domains]
        vol_router_weights: InputTensor[dtype = DType.float32, rank=3],         # [1, 8, num_domains]
        vol_router_bias: InputTensor[dtype = DType.float32, rank=2],            # [8, num_domains]
        risk_router_weights: InputTensor[dtype = DType.float32, rank=3],        # [1, 8, num_domains]
        risk_router_bias: InputTensor[dtype = DType.float32, rank=2],           # [8, num_domains]
        stats_router_weights: InputTensor[dtype = DType.float32, rank=3],       # [3, 12, num_domains]
        stats_router_bias: InputTensor[dtype = DType.float32, rank=2],          # [12, num_domains]
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn compute_all_routing_logits[
            width: Int
        ](idx: IndexList[base_logits.rank]) -> SIMD[DType.float32, width]:
            var b = idx[0]
            var t = idx[1]
            var d = idx[2]

            # ===== BASE LOGITS (Domain Classifier) =====
            var base_accumulator = stack_allocation[SIMD_WIDTH, Float32]()
            var base_bias_val = domain_classifier_bias.load[1](IndexList[1](d))[0]

            @parameter
            for i in range(SIMD_WIDTH):
                base_accumulator[i] = base_bias_val / Float32(SIMD_WIDTH)

            # Vectorized computation for base logits
            @parameter
            fn base_computation[v_width: Int](i: Int):
                if i + v_width <= HIDDEN_SIZE * 2:
                    var feat_vals = extracted_features.load[v_width](IndexList[3](b, t, i))
                    var weight_vals = domain_classifier_weights.load[v_width](IndexList[2](i, d))

                    @parameter
                    for j in range(v_width):
                        var lane_idx = j % SIMD_WIDTH
                        base_accumulator[lane_idx] += feat_vals[j] * weight_vals[j]
                else:
                    for j in range(i, HIDDEN_SIZE * 2):
                        var feat_val = extracted_features.load[1](IndexList[3](b, t, j))[0]
                        var weight_val = domain_classifier_weights.load[1](IndexList[2](j, d))[0]
                        base_accumulator[0] += feat_val * weight_val

            vectorize[base_computation, SIMD_WIDTH](HIDDEN_SIZE * 2)

            var base_result: Float32 = 0.0
            @parameter
            for i in range(SIMD_WIDTH):
                base_result += base_accumulator[i]

            # ===== VOLATILITY LOGITS =====
            var vol_val = market_volatility.load[1](IndexList[3](b, t, 0))[0]
            var vol_result: Float32 = 0.0

            # Two-layer volatility router: vol -> hidden[8] -> domains
            var vol_hidden = stack_allocation[8, Float32]()
            for h in range(8):
                var vol_w1 = vol_router_weights.load[1](IndexList[3](0, h, d))[0]
                var vol_b1 = vol_router_bias.load[1](IndexList[2](h, d))[0]
                vol_hidden[h] = max(vol_val * vol_w1 + vol_b1, Float32(0.0))  # ReLU
                vol_result += vol_hidden[h]
            vol_result /= 8.0  # Average pooling

            # ===== RISK LOGITS =====
            var risk_val = risk_factors.load[1](IndexList[3](b, t, 0))[0]
            var risk_result: Float32 = 0.0

            # Two-layer risk router
            var risk_hidden = stack_allocation[8, Float32]()
            for h in range(8):
                var risk_w1 = risk_router_weights.load[1](IndexList[3](0, h, d))[0]
                var risk_b1 = risk_router_bias.load[1](IndexList[2](h, d))[0]
                risk_hidden[h] = max(risk_val * risk_w1 + risk_b1, Float32(0.0))  # ReLU
                risk_result += risk_hidden[h]
            risk_result /= 8.0  # Average pooling

            # ===== STATS LOGITS =====
            var stats_result: Float32 = 0.0

            # Two-layer stats router: stats[3] -> hidden[12] -> domains
            var stats_hidden = stack_allocation[12, Float32]()
            for h in range(12):
                var hidden_val: Float32 = 0.0
                for s in range(3):
                    var stat_val = embedding_stats.load[1](IndexList[3](b, t, s))[0]
                    var stat_w = stats_router_weights.load[1](IndexList[3](s, h, d))[0]
                    hidden_val += stat_val * stat_w
                var stat_b = stats_router_bias.load[1](IndexList[2](h, d))[0]
                stats_hidden[h] = max(hidden_val + stat_b, Float32(0.0))  # ReLU
                stats_result += stats_hidden[h]
            stats_result /= 12.0  # Average pooling

            # Store results in respective output tensors
            base_logits.store[1](IndexList[3](b, t, d), base_result)
            vol_logits.store[1](IndexList[3](b, t, d), vol_result)
            risk_logits.store[1](IndexList[3](b, t, d), risk_result)
            stats_logits.store[1](IndexList[3](b, t, d), stats_result)

            return SIMD[DType.float32, width](base_result)  # Return dummy value for foreach

        foreach[
            compute_all_routing_logits,
            target=target,
            simd_width=1
        ](base_logits, ctx)


@register("mojo_master_router")
struct MojoMasterRouter:
    """Master routing kernel that combines all signals and makes final decisions."""
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        domain_assignments: OutputTensor[dtype = DType.int32, rank=2],
        routing_probs: OutputTensor[dtype = DType.float32, rank=3],
        routing_logits: OutputTensor[dtype = DType.float32, rank=3],
        base_logits: InputTensor[dtype = DType.float32, rank=3],
        vol_logits: InputTensor[dtype = DType.float32, rank=3],
        risk_logits: InputTensor[dtype = DType.float32, rank=3],
        stats_logits: InputTensor[dtype = DType.float32, rank=3],
        market_volatility: InputTensor[dtype = DType.float32, rank=3],
        embedding_stats: InputTensor[dtype = DType.float32, rank=3],
        is_training: InputTensor[dtype = DType.bool, rank=0],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn master_routing_decision[
            simd_width: Int
        ](idx: IndexList[domain_assignments.rank]) -> SIMD[DType.int32, simd_width]:
            var b = idx[0]
            var t = idx[1]

            var combined_logits = stack_allocation[NUM_DOMAINS, Float32]()
            var domain_probs = stack_allocation[NUM_DOMAINS, Float32]()

            # Get market conditions for enhanced routing
            var volatility = market_volatility.load[1](IndexList[3](b, t, 0))[0]
            var emb_mean = embedding_stats.load[1](IndexList[3](b, t, 0))[0]
            var emb_var = embedding_stats.load[1](IndexList[3](b, t, 1))[0]
            var emb_abs_mean = embedding_stats.load[1](IndexList[3](b, t, 2))[0]

            # Clamp values for stability
            volatility = max(min(volatility, Float32(0.1)), Float32(0.0001))

            var best_domain: Int32 = 0
            var best_score: Float32 = -1e6

            # ===== ADVANCED LOGIT COMBINATION =====
            for d in range(NUM_DOMAINS):
                var base_logit = base_logits.load[1](IndexList[3](b, t, d))[0]
                var vol_logit = vol_logits.load[1](IndexList[3](b, t, d))[0]
                var risk_logit = risk_logits.load[1](IndexList[3](b, t, d))[0]
                var stats_logit = stats_logits.load[1](IndexList[3](b, t, d))[0]

                # Adaptive weighting based on market conditions
                var base_weight: Float32 = 1.0
                var vol_weight: Float32 = 0.8
                var risk_weight: Float32 = 0.6
                var stats_weight: Float32 = 0.4

                # Increase volatility importance during high-vol periods
                if volatility > 0.025:
                    vol_weight = 1.2
                    risk_weight = 1.0

                # Increase stats importance for extreme embeddings
                if emb_var > 2.0 or abs(emb_mean) > 1.0:
                    stats_weight = 0.8

                # Weighted combination
                var combined_logit = (base_logit * base_weight +
                    vol_logit * vol_weight +
                    risk_logit * risk_weight +
                    stats_logit * stats_weight) / (base_weight + vol_weight + risk_weight + stats_weight)

                # Enhanced domain-specific boosts
                var domain_boost: Float32 = 0.0
                if d == 0:  # Equities
                    if volatility > 0.015 and volatility < 0.04 and emb_var > 0.3 and emb_var < 2.0:
                        domain_boost = 1.0
                elif d == 1:  # Fixed Income
                    if volatility < 0.005 and emb_var < 0.2:
                        domain_boost = 1.2
                elif d == 2:  # Commodities
                    if volatility > 0.01 and volatility < 0.03 and emb_abs_mean > 0.5:
                        domain_boost = 0.8
                elif d == 3:  # FX
                    if abs(emb_mean) > 1.0 and volatility < 0.02:
                        domain_boost = 1.0
                elif d == 4:  # Derivatives
                    if volatility > 0.035 and emb_var > 2.0:
                        domain_boost = 1.5
                elif d == 5:  # Credit
                    if volatility < 0.01 and emb_var < 0.5 and abs(emb_mean) < 0.5:
                        domain_boost = 0.8

                combined_logit += domain_boost

                # Temperature scaling for training vs inference
                if is_training.load[1](IndexList[0]())[0]:
                    combined_logit /= 0.9  # Slightly lower for more confident training decisions
                else:
                    combined_logit /= 0.8  # Even lower for inference
                combined_logit = max(min(combined_logit, MAX_LOGIT_VALUE), MIN_LOGIT_VALUE)

                combined_logits[d] = combined_logit
                routing_logits.store[1](IndexList[3](b, t, d), combined_logit)

                if combined_logit > best_score:
                    best_score = combined_logit
                    best_domain = d

                    # ===== SOFTMAX WITH NUMERICAL STABILITY =====
            var total_exp: Float32 = 0.0
            for d in range(NUM_DOMAINS):
                var exp_logit = exp(combined_logits[d] - best_score)
                domain_probs[d] = exp_logit
                total_exp += exp_logit

            var total_exp_safe = max(total_exp, NUMERICAL_EPSILON)
            var max_prob: Float32 = 0.0

            for d in range(NUM_DOMAINS):
                var prob = domain_probs[d] / total_exp_safe
                routing_probs.store[1](IndexList[3](b, t, d), prob)
                if Int32(d) == best_domain:
                    max_prob = prob

                    # ===== CONFIDENCE-BASED FALLBACK =====
            if max_prob < 0.4:  # Slightly higher threshold for better decisions
                # Multi-factor fallback logic
                if volatility > 0.04 and emb_var > 3.0:
                    best_domain = Int32(4)  # Derivatives - very high vol + variance
                elif volatility < 0.003 and emb_var < 0.1:
                    best_domain = Int32(1)  # Fixed Income - very low vol + variance
                elif volatility > 0.02 and volatility < 0.035 and emb_var > 0.8:
                    best_domain = Int32(0)  # Equities - mid-high vol + variance
                elif abs(emb_mean) > 1.5 and volatility < 0.015:
                    best_domain = Int32(3)  # FX - strong trend + low-mid vol
                elif volatility > 0.012 and volatility < 0.025 and emb_abs_mean > 1.0:
                    best_domain = Int32(2)  # Commodities - seasonal patterns
                else:
                    best_domain = Int32(5)  # Credit - default for unclear cases

            return SIMD[DType.int32, simd_width](best_domain)

        foreach[
            master_routing_decision,
            target=target,
            simd_width=1
        ](domain_assignments, ctx)


@register("mojo_expert_computation")
struct MojoExpertComputation:
    """Expert network computation with domain-specific processing."""
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        expert_outputs: OutputTensor[dtype = DType.float32, rank=3],      # [batch, seq, hidden]
        extracted_features: InputTensor[dtype = DType.float32, rank=3],   # [batch, seq, hidden*2]
        domain_assignments: InputTensor[dtype = DType.int32, rank=2],     # [batch, seq]
        expert_weights: InputTensor[dtype = DType.float32, rank=3],       # [num_domains, hidden*2, hidden]
        expert_bias: InputTensor[dtype = DType.float32, rank=2],          # [num_domains, hidden]
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn compute_expert_output[
            width: Int
        ](idx: IndexList[expert_outputs.rank]) -> SIMD[DType.float32, width]:
            var b = idx[0]
            var t = idx[1]
            var h = idx[2]  # output hidden dimension

            # Get assigned domain for this timestep
            var assigned_domain = domain_assignments.load[1](IndexList[2](b, t))[0]
            var domain_idx = Int(assigned_domain)

            # Ensure domain index is valid
            if domain_idx < 0 or domain_idx >= NUM_DOMAINS:
                domain_idx = 0  # Fallback to first domain

            # Initialize with bias
            var result = expert_bias.load[1](IndexList[2](domain_idx, h))[0]

            # Compute expert-specific transformation
            var accumulator = stack_allocation[SIMD_WIDTH, Float32]()
            @parameter
            for i in range(SIMD_WIDTH):
                accumulator[i] = 0.0

            # Vectorized computation
            @parameter
            fn expert_matmul[v_width: Int](i: Int):
                if i + v_width <= HIDDEN_SIZE * 2:
                    var feat_vals = extracted_features.load[v_width](IndexList[3](b, t, i))
                    var weight_vals = expert_weights.load[v_width](IndexList[3](domain_idx, i, h))

                    @parameter
                    for j in range(v_width):
                        var lane_idx = j % SIMD_WIDTH
                        accumulator[lane_idx] += feat_vals[j] * weight_vals[j]
                else:
                    for j in range(i, HIDDEN_SIZE * 2):
                        var feat_val = extracted_features.load[1](IndexList[3](b, t, j))[0]
                        var weight_val = expert_weights.load[1](IndexList[3](domain_idx, j, h))[0]
                        accumulator[0] += feat_val * weight_val

            vectorize[expert_matmul, SIMD_WIDTH](HIDDEN_SIZE * 2)

            # Reduce accumulator
            @parameter
            for i in range(SIMD_WIDTH):
                result += accumulator[i]

            # Apply domain-specific activation
            if domain_idx == 4:  # Derivatives - use tanh for bounded output
                result = tanh(result)
            elif domain_idx == 1:  # Fixed Income - gentle activation
                result = result * 0.8
            else:  # Others - standard ReLU
                result = max(result, Float32(0.0))

            return SIMD[DType.float32, width](result)

        foreach[
            compute_expert_output,
            target=target,
            simd_width=1
        ](expert_outputs, ctx)


@register("mojo_routing_loss")
struct MojoRoutingLoss:
    """Mojo-powered loss computation for better training integration."""
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        loss_output: OutputTensor[dtype = DType.float32, rank=1],          # [1] - scalar loss
        routing_logits: InputTensor[dtype = DType.float32, rank=3],        # [batch, seq, num_domains]
        targets: InputTensor[dtype = DType.int32, rank=2],                 # [batch, seq]
        routing_probs: InputTensor[dtype = DType.float32, rank=3],         # [batch, seq, num_domains]
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn compute_loss[
            width: Int
        ](idx: IndexList[loss_output.rank]) -> SIMD[DType.float32, width]:
            var total_loss: Float32 = 0.0
            var total_elements: Float32 = 0.0

            var batch_size = routing_logits.shape()[0]
            var seq_len = routing_logits.shape()[1]

            # Compute cross-entropy loss with label smoothing
            for b in range(batch_size):
                for t in range(seq_len):
                    var target_domain = targets.load[1](IndexList[2](b, t))[0]
                    var target_idx = Int(target_domain)

                    # Ensure valid target
                    if target_idx < 0 or target_idx >= NUM_DOMAINS:
                        continue

                    # Compute log-softmax manually for numerical stability
                    var max_logit: Float32 = -1e6
                    for d in range(NUM_DOMAINS):
                        var logit = routing_logits.load[1](IndexList[3](b, t, d))[0]
                        if logit > max_logit:
                            max_logit = logit

                    var log_sum_exp: Float32 = 0.0
                    for d in range(NUM_DOMAINS):
                        var logit = routing_logits.load[1](IndexList[3](b, t, d))[0]
                        log_sum_exp += exp(logit - max_logit)

                    # Fixed: Use explicit type conversion and proper Float32 arithmetic
                    var log_softmax_target = routing_logits.load[1](IndexList[3](b, t, target_idx))[0] - max_logit - log(Float32(log_sum_exp))

                    # Label smoothing (smoothing factor = 0.05) - Fixed type issues
                    var smoothed_loss = Float32(-0.95) * log_softmax_target

                    # Add uniform distribution component
                    var uniform_component: Float32 = 0.0
                    for d in range(NUM_DOMAINS):
                        var logit = routing_logits.load[1](IndexList[3](b, t, d))[0]
                        var log_softmax_d = logit - max_logit - log(Float32(log_sum_exp))
                        uniform_component += log_softmax_d
                    uniform_component *= Float32(-0.05) / Float32(NUM_DOMAINS)

                    smoothed_loss += uniform_component
                    total_loss += smoothed_loss
                    total_elements += 1.0

                    # Average loss
            var avg_loss = total_loss / max(total_elements, Float32(1.0))

            return SIMD[DType.float32, width](avg_loss)

        foreach[
            compute_loss,
            target=target,
            simd_width=1
        ](loss_output, ctx)


@register("mojo_gradient_computation")
struct MojoGradientComputation:
    """Advanced gradient computation with domain-aware adjustments."""
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        grad_output: OutputTensor[dtype = DType.float32, rank=3],          # [batch, seq, num_domains]
        routing_probs: InputTensor[dtype = DType.float32, rank=3],         # [batch, seq, num_domains]
        targets: InputTensor[dtype = DType.int32, rank=2],                 # [batch, seq]
        market_volatility: InputTensor[dtype = DType.float32, rank=3],     # [batch, seq, 1]
        embedding_stats: InputTensor[dtype = DType.float32, rank=3],       # [batch, seq, 3]
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn compute_gradients[
            width: Int
        ](idx: IndexList[grad_output.rank]) -> SIMD[DType.float32, width]:
            var b = idx[0]
            var t = idx[1]
            var d = idx[2]

            var prob = routing_probs.load[1](IndexList[3](b, t, d))[0]
            var target_domain = targets.load[1](IndexList[2](b, t))[0]
            var volatility = market_volatility.load[1](IndexList[3](b, t, 0))[0]
            var emb_var = embedding_stats.load[1](IndexList[3](b, t, 1))[0]

            # Standard cross-entropy gradient
            var grad: Float32
            if Int32(d) == target_domain:
                grad = prob - 1.0
            else:
                grad = prob

            # Domain-specific gradient scaling for better learning
            var scale_factor: Float32 = 1.0
            if d == 1 and volatility < 0.005 and emb_var < 0.2:  # Fixed Income
                scale_factor = 1.3  # Boost learning for low-vol scenarios
            elif d == 4 and volatility > 0.03 and emb_var > 2.0:  # Derivatives
                scale_factor = 1.3  # Boost learning for high-vol scenarios
            elif d == 0 and volatility > 0.015 and volatility < 0.04:  # Equities
                scale_factor = 1.15  # Moderate boost for mid-vol

            grad *= scale_factor

            return SIMD[DType.float32, width](grad)

        foreach[
            compute_gradients,
            target=target,
            simd_width=1
        ](grad_output, ctx)