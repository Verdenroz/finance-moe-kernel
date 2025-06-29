from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from math import exp, sqrt, tanh, log
from builtin.math import abs, max, min
from memory import stack_allocation
from algorithm import vectorize, parallelize
from sys import simdwidthof

alias HIDDEN_SIZE = 32
alias NUM_DOMAINS = 4
alias VECTORIZATION_WIDTH = 8
alias MAX_LOGIT_VALUE = 15.0
alias MIN_LOGIT_VALUE = -15.0
alias NUMERICAL_EPSILON = 1e-7
alias SIMD_WIDTH = 8
alias TILE_SIZE = 32
alias PARALLEL_WORK_SIZE = 4


@register("feature_extractor")
struct FeatureExtractor:
    """Feature extraction replacing PyTorch Sequential layers."""
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

            # Matrix multiply with ReLU activation
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


@register("multi_router")
struct MultiRouter:
    """Single kernel handling ALL router computations."""
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


@register("routing_engine")
struct RoutingEngine:
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
        ctx: DeviceContextPtr,
    ) raises:

        @parameter
        @always_inline
        fn route[
            simd_width: Int
        ](idx: IndexList[domain_assignments.rank]) -> SIMD[DType.int32, simd_width]:
            var b = idx[0]
            var t = idx[1]

            var logits = stack_allocation[NUM_DOMAINS, Float32]()
            var max_logit: Float32 = -1e6
            var best_domain: Int32 = 0

            # --- weighted sum (constant weights — can be learnt in PyTorch) ----
            for d in range(NUM_DOMAINS):
                var l = 1.20 * base_logits.load[1](IndexList[3](b, t, d))[0] +   \
                        1.00 * vol_logits .load[1](IndexList[3](b, t, d))[0] +   \
                        0.80 * risk_logits.load[1](IndexList[3](b, t, d))[0] +   \
                        0.60 * stats_logits.load[1](IndexList[3](b, t, d))[0]

                logits[d] = l
                if l > max_logit:
                    max_logit  = l
                    best_domain = Int32(d)

            # -- soft-max --------------------------------------------------------
            var exp_sum: Float32 = 0.0
            for d in range(NUM_DOMAINS):
                logits[d] = exp(logits[d] - max_logit)
                exp_sum  += logits[d]

            for d in range(NUM_DOMAINS):
                var p = logits[d] / max(exp_sum, 1e-8)
                routing_probs.store[1](IndexList[3](b, t, d), p)
                routing_logits.store[1](IndexList[3](b, t, d), log(p + 1e-9))

            return SIMD[DType.int32, simd_width](best_domain)

        foreach[
            route,
            target      = target,
            simd_width  = 1
        ](domain_assignments, ctx)



@register("expert_processor")
struct ExpertProcessor:
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
            if domain_idx == 3:  # Derivatives - use tanh for bounded output
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


@register("routing_loss")
struct RoutingLoss:
    """Loss computation for better training integration."""
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


@register("gradient_engine")
struct GradientEngine:
    """Gradient computation with domain-aware adjustments."""
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
            elif d == 3 and volatility > 0.03 and emb_var > 2.0:  # Derivatives
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