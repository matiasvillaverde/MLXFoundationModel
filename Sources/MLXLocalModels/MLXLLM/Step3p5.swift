import Foundation
import MLX
import MLXFast
import MLXNN

private let step3p5FullAttention = "full_attention"
private let step3p5SlidingAttention = "sliding_attention"

private func step3p5SwiGLU(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    silu(gate) * up
}

private func step3p5ClampedSwiGLU(
    up: MLXArray,
    gate: MLXArray,
    limit: Float
) -> MLXArray {
    let gateLimit = MLXArray(limit, dtype: gate.dtype)
    let upLimit = MLXArray(limit, dtype: up.dtype)
    return clip(silu(gate), max: gateLimit)
        * clip(up, min: -upLimit, max: upLimit)
}

internal struct Step3p5AttentionOverride: Codable, Equatable, Sendable {
    var attentionHeads: Int
    var attentionGroups: Int

    private enum CodingKeys: String, CodingKey {
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
    }
}

internal struct Step3p5Configuration: Codable, Equatable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var vocabularySize: Int
    var attentionHeads: Int
    var attentionGroups: Int
    var headDim: Int
    var intermediateSize: Int
    var rmsNormEps: Float
    var ropeTheta: StringOrNumber
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var layerTypes: [String]?
    var yarnOnlyTypes: [String]?
    var partialRotaryFactors: [Float]?
    var attentionOverride: Step3p5AttentionOverride?
    var usesHeadWiseAttentionGate: Bool
    var expertCount: Int
    var expertsPerToken: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var moeLayers: String?
    var routerScaling: Float
    var normalizesExpertWeights: Bool
    var swigluLimits: [Float?]?
    var sharedSwigluLimits: [Float?]?
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "step3p5",
        hiddenSize: Int,
        hiddenLayers: Int,
        vocabularySize: Int,
        attentionHeads: Int,
        attentionGroups: Int,
        headDim: Int,
        intermediateSize: Int,
        rmsNormEps: Float = 1e-5,
        ropeTheta: StringOrNumber = .float(10_000),
        ropeScaling: [String: StringOrNumber]? = nil,
        maxPositionEmbeddings: Int = 262_144,
        slidingWindow: Int = 512,
        layerTypes: [String]? = nil,
        yarnOnlyTypes: [String]? = nil,
        partialRotaryFactors: [Float]? = nil,
        attentionOverride: Step3p5AttentionOverride? = nil,
        usesHeadWiseAttentionGate: Bool = true,
        expertCount: Int = 288,
        expertsPerToken: Int = 8,
        moeIntermediateSize: Int = 1_280,
        sharedExpertIntermediateSize: Int = 1_280,
        moeLayers: String? = nil,
        routerScaling: Float = 3,
        normalizesExpertWeights: Bool = true,
        swigluLimits: [Float?]? = nil,
        sharedSwigluLimits: [Float?]? = nil,
        tieWordEmbeddings: Bool = false
    ) {
        precondition(hiddenSize > 0, "Step3.5 hidden_size must be positive")
        precondition(hiddenLayers > 0, "Step3.5 num_hidden_layers must be positive")
        precondition(vocabularySize > 0, "Step3.5 vocab_size must be positive")
        precondition(attentionHeads > 0, "Step3.5 num_attention_heads must be positive")
        precondition(attentionGroups > 0, "Step3.5 num_attention_groups must be positive")
        precondition(headDim > 0, "Step3.5 head_dim must be positive")
        precondition(intermediateSize > 0, "Step3.5 intermediate_size must be positive")
        precondition(slidingWindow > 0, "Step3.5 sliding_window must be positive")
        precondition(expertCount > 0, "Step3.5 moe_num_experts must be positive")
        precondition(expertsPerToken > 0, "Step3.5 moe_top_k must be positive")
        precondition(
            expertsPerToken <= expertCount,
            "Step3.5 moe_top_k cannot exceed moe_num_experts"
        )

        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.vocabularySize = vocabularySize
        self.attentionHeads = attentionHeads
        self.attentionGroups = attentionGroups
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
        self.yarnOnlyTypes = yarnOnlyTypes
        self.partialRotaryFactors = partialRotaryFactors
        self.attentionOverride = attentionOverride
        self.usesHeadWiseAttentionGate = usesHeadWiseAttentionGate
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.moeIntermediateSize = moeIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.moeLayers = moeLayers
        self.routerScaling = routerScaling
        self.normalizesExpertWeights = normalizesExpertWeights
        self.swigluLimits = swigluLimits
        self.sharedSwigluLimits = sharedSwigluLimits
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    func layerType(at layerIndex: Int) -> String {
        if let layerTypes, layerIndex < layerTypes.count {
            return layerTypes[layerIndex]
        }
        return layerIndex.isMultiple(of: 2) ? step3p5SlidingAttention : step3p5FullAttention
    }

    func ropeLayerType(at layerIndex: Int) -> String {
        if let layerTypes, layerIndex < layerTypes.count {
            return layerTypes[layerIndex]
        }
        return step3p5FullAttention
    }

    func ropeTheta(at layerIndex: Int) -> Float {
        if let values = ropeTheta.asFloats() {
            if layerIndex < values.count {
                return values[layerIndex]
            }
            if values.count == 1, let first = values.first {
                return first
            }
        }
        return ropeTheta.asFloat() ?? 10_000
    }

    func partialRotaryFactor(at layerIndex: Int) -> Float {
        guard let partialRotaryFactors, layerIndex < partialRotaryFactors.count else {
            return 1
        }
        return partialRotaryFactors[layerIndex]
    }

    func swigluLimit(at layerIndex: Int) -> Float? {
        Self.limit(from: swigluLimits, layerIndex: layerIndex)
    }

    func sharedSwigluLimit(at layerIndex: Int) -> Float? {
        Self.limit(from: sharedSwigluLimits, layerIndex: layerIndex)
    }

    func ropeScaling(at layerIndex: Int) -> [String: StringOrNumber]? {
        let ropeLayerType = ropeLayerType(at: layerIndex)
        if let yarnOnlyTypes, !yarnOnlyTypes.isEmpty, !yarnOnlyTypes.contains(ropeLayerType) {
            return nil
        }
        return ropeScaling
    }

    private static func limit(from limits: [Float?]?, layerIndex: Int) -> Float? {
        guard let limits, layerIndex < limits.count, let value = limits[layerIndex], value > 0 else {
            return nil
        }
        return value
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case yarnOnlyTypes = "yarn_only_types"
        case partialRotaryFactors = "partial_rotary_factors"
        case attentionOverride = "attention_other_setting"
        case usesHeadWiseAttentionGate = "use_head_wise_attn_gate"
        case expertCount = "moe_num_experts"
        case expertsPerToken = "moe_top_k"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "share_expert_dim"
        case moeLayers = "moe_layers_enum"
        case routerScaling = "moe_router_scaling_factor"
        case normalizesExpertWeights = "norm_expert_weight"
        case swigluLimits = "swiglu_limits"
        case sharedSwigluLimits = "swiglu_limits_shared"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "step3p5",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            attentionGroups: try container.decode(Int.self, forKey: .attentionGroups),
            headDim: try container.decode(Int.self, forKey: .headDim),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            ropeTheta: try container.decodeIfPresent(StringOrNumber.self, forKey: .ropeTheta)
                ?? .float(10_000),
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 262_144,
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512,
            layerTypes: try container.decodeIfPresent([String].self, forKey: .layerTypes),
            yarnOnlyTypes: try container.decodeIfPresent([String].self, forKey: .yarnOnlyTypes),
            partialRotaryFactors: try container.decodeIfPresent(
                [Float].self,
                forKey: .partialRotaryFactors
            ),
            attentionOverride: try container.decodeIfPresent(
                Step3p5AttentionOverride.self,
                forKey: .attentionOverride
            ),
            usesHeadWiseAttentionGate: try container.decodeIfPresent(
                Bool.self,
                forKey: .usesHeadWiseAttentionGate
            ) ?? true,
            expertCount: try container.decodeIfPresent(Int.self, forKey: .expertCount) ?? 288,
            expertsPerToken: try container.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 8,
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ) ?? 1_280,
            sharedExpertIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .sharedExpertIntermediateSize
            ) ?? 1_280,
            moeLayers: try container.decodeIfPresent(String.self, forKey: .moeLayers),
            routerScaling: try container.decodeIfPresent(Float.self, forKey: .routerScaling) ?? 3,
            normalizesExpertWeights: try container.decodeIfPresent(
                Bool.self,
                forKey: .normalizesExpertWeights
            ) ?? true,
            swigluLimits: try container.decodeIfPresent([Float?].self, forKey: .swigluLimits),
            sharedSwigluLimits: try container.decodeIfPresent(
                [Float?].self,
                forKey: .sharedSwigluLimits
            ),
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? false
        )
    }
}

internal struct Step3p5LayerPlan: Equatable, Sendable {
    let layerTypes: [String]
    let moeLayerIndices: Set<Int>
    let firstSlidingLayerIndex: Int?
    let firstFullLayerIndex: Int?

    init(_ configuration: Step3p5Configuration) {
        self.layerTypes = (0 ..< configuration.hiddenLayers).map {
            configuration.layerType(at: $0)
        }
        self.moeLayerIndices = Self.parseMoELayers(
            configuration.moeLayers,
            layerCount: configuration.hiddenLayers
        )
        self.firstSlidingLayerIndex = layerTypes.firstIndex(of: step3p5SlidingAttention)
        self.firstFullLayerIndex = layerTypes.firstIndex { $0 != step3p5SlidingAttention }
    }

    func isSlidingLayer(_ layerIndex: Int) -> Bool {
        layerTypes[layerIndex] == step3p5SlidingAttention
    }

    func isMoELayer(_ layerIndex: Int) -> Bool {
        moeLayerIndices.contains(layerIndex)
    }

    private static func parseMoELayers(_ value: String?, layerCount: Int) -> Set<Int> {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return Set(1 ..< layerCount)
        }
        return Set(
            value
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                .filter { $0 >= 0 && $0 < layerCount }
        )
    }
}

private struct Step3p5AttentionLayout: Sendable, Equatable {
    let queryHeads: Int
    let keyValueHeads: Int
    let headDim: Int
    let queryProjectionSize: Int
    let keyValueProjectionSize: Int
    let rotaryDimensions: Int
    let scale: Float

    init(_ config: Step3p5Configuration, layerIndex: Int) {
        let plan = Step3p5LayerPlan(config)
        if plan.isSlidingLayer(layerIndex), let override = config.attentionOverride {
            self.queryHeads = override.attentionHeads
            self.keyValueHeads = override.attentionGroups
        } else {
            self.queryHeads = config.attentionHeads
            self.keyValueHeads = config.attentionGroups
        }
        precondition(queryHeads > 0, "Step3.5 query head count must be positive")
        precondition(keyValueHeads > 0, "Step3.5 KV head count must be positive")
        self.headDim = config.headDim
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        let rotaryDimensions = Int(Float(headDim) * config.partialRotaryFactor(at: layerIndex))
        precondition(rotaryDimensions > 0, "Step3.5 rotary dimensions must be positive")
        precondition(rotaryDimensions.isMultiple(of: 2), "Step3.5 rotary dimensions must be even")
        self.rotaryDimensions = rotaryDimensions
        self.scale = pow(Float(headDim), -0.5)
    }
}

private final class Step3p5MLP: Module, UnaryLayer {
    private let limit: Float?

    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(hiddenSize: Int, intermediateSize: Int, limit: Float? = nil) {
        self.limit = limit
        _gateProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let gate = gateProjection(input)
        let up = upProjection(input)
        if let limit {
            return downProjection(step3p5ClampedSwiGLU(up: up, gate: gate, limit: limit))
        }
        return downProjection(step3p5SwiGLU(gate, up))
    }
}

private final class Step3p5MoEGate: Module {
    private let topK: Int
    private let routerScaling: Float
    private let normalizesExpertWeights: Bool

    @ModuleInfo(key: "gate") private var gate: Linear
    @ParameterInfo(key: "router_bias") private var routerBias: MLXArray

    init(_ config: Step3p5Configuration) {
        self.topK = config.expertsPerToken
        self.routerScaling = config.routerScaling
        self.normalizesExpertWeights = config.normalizesExpertWeights
        _gate.wrappedValue = Linear(config.hiddenSize, config.expertCount, bias: false)
        _routerBias.wrappedValue = MLXArray.zeros([config.expertCount])
    }

    func callAsFunction(_ input: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let scores = sigmoid(gate(input).asType(.float32))
        let correctedScores = scores + routerBias.asType(scores.dtype)
        let indices = argPartition(-correctedScores, kth: topK - 1, axis: -1)[
            .ellipsis, ..<topK
        ]
        var weights = takeAlong(scores, indices, axis: -1)
        if normalizesExpertWeights {
            weights = weights / (sum(weights, axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * routerScaling)
    }
}

private final class Step3p5ExpertGLU: Module {
    private let limit: Float?

    @ModuleInfo(key: "gate_proj") private var gateProjection: SwitchLinear
    @ModuleInfo(key: "up_proj") private var upProjection: SwitchLinear
    @ModuleInfo(key: "down_proj") private var downProjection: SwitchLinear

    init(
        inputDimensions: Int,
        hiddenDimensions: Int,
        expertCount: Int,
        limit: Float? = nil
    ) {
        self.limit = limit
        _gateProjection.wrappedValue = SwitchLinear(
            inputDims: inputDimensions,
            outputDims: hiddenDimensions,
            numExperts: expertCount,
            bias: false
        )
        _upProjection.wrappedValue = SwitchLinear(
            inputDims: inputDimensions,
            outputDims: hiddenDimensions,
            numExperts: expertCount,
            bias: false
        )
        _downProjection.wrappedValue = SwitchLinear(
            inputDims: hiddenDimensions,
            outputDims: inputDimensions,
            numExperts: expertCount,
            bias: false
        )
    }

    func callAsFunction(_ input: MLXArray, expertIndices: MLXArray) -> MLXArray {
        let routedInput = expandedDimensions(input, axes: [-2, -3])
        let sortedDispatch = SwitchExpertDispatch.shouldSort(expertIndices: expertIndices)
        let permutation = sortedDispatch
            ? SwitchExpertPermutation(input: routedInput, expertIndices: expertIndices)
            : nil

        let expertInput = permutation?.sortedInput ?? routedInput
        let selectedExpertIndices = permutation?.sortedExpertIndices ?? expertIndices
        let gate = gateProjection(
            expertInput,
            selectedExpertIndices,
            sortedIndices: sortedDispatch
        )
        let up = upProjection(
            expertInput,
            selectedExpertIndices,
            sortedIndices: sortedDispatch
        )
        let activated = if let limit {
            step3p5ClampedSwiGLU(up: up, gate: gate, limit: limit)
        } else {
            step3p5SwiGLU(gate, up)
        }
        let output = downProjection(
            activated,
            selectedExpertIndices,
            sortedIndices: sortedDispatch
        )

        return squeezed(permutation?.restore(output) ?? output, axis: -2)
    }
}

private final class Step3p5MoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") private var gate: Step3p5MoEGate
    @ModuleInfo(key: "switch_mlp") private var switchMLP: Step3p5ExpertGLU
    @ModuleInfo(key: "share_expert") fileprivate var sharedExpert: Step3p5MLP

    init(_ config: Step3p5Configuration, layerIndex: Int) {
        _gate.wrappedValue = Step3p5MoEGate(config)
        _switchMLP.wrappedValue = Step3p5ExpertGLU(
            inputDimensions: config.hiddenSize,
            hiddenDimensions: config.moeIntermediateSize,
            expertCount: config.expertCount,
            limit: config.swigluLimit(at: layerIndex)
        )
        _sharedExpert.wrappedValue = Step3p5MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.sharedExpertIntermediateSize,
            limit: config.sharedSwigluLimit(at: layerIndex)
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (indices, weights) = gate(input)
        let routed = switchMLP(input, expertIndices: indices)
        return (routed * weights[.ellipsis, .newAxis]).sum(axis: -2)
            + sharedExpert(input)
    }
}

private final class Step3p5Attention: Module {
    fileprivate let layout: Step3p5AttentionLayout
    private let usesHeadWiseAttentionGate: Bool
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") private var keyNorm: RMSNorm
    @ModuleInfo(key: "g_proj") private var gateProjection: Linear?

    init(_ config: Step3p5Configuration, layerIndex: Int) {
        self.layout = Step3p5AttentionLayout(config, layerIndex: layerIndex)
        self.usesHeadWiseAttentionGate = config.usesHeadWiseAttentionGate
        self.rope = initializeRope(
            dims: layout.rotaryDimensions,
            base: config.ropeTheta(at: layerIndex),
            traditional: false,
            scalingConfig: config.ropeScaling(at: layerIndex),
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: false
        )
        _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
        if config.usesHeadWiseAttentionGate {
            _gateProjection.wrappedValue = Linear(
                config.hiddenSize,
                layout.queryHeads,
                bias: false
            )
        }
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)

        var queries = queryNorm(
            queryProjection(input)
                .reshaped(batchSize, sequenceLength, layout.queryHeads, layout.headDim)
        )
        .transposed(0, 2, 1, 3)
        var keys = keyNorm(
            keyProjection(input)
                .reshaped(batchSize, sequenceLength, layout.keyValueHeads, layout.headDim)
        )
        .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, sequenceLength, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)

        if usesHeadWiseAttentionGate, let gateProjection {
            output = output * sigmoid(gateProjection(input))[.ellipsis, .newAxis]
        }

        return outputProjection(output.reshaped(batchSize, sequenceLength, -1))
    }
}

private final class Step3p5DecoderLayer: Module {
    let isSliding: Bool
    let isMoE: Bool

    @ModuleInfo(key: "self_attn") fileprivate var attention: Step3p5Attention
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm
    fileprivate let mlp: UnaryLayer

    init(_ config: Step3p5Configuration, layerIndex: Int, plan: Step3p5LayerPlan) {
        self.isSliding = plan.isSlidingLayer(layerIndex)
        self.isMoE = plan.isMoELayer(layerIndex)
        _attention.wrappedValue = Step3p5Attention(config, layerIndex: layerIndex)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self.mlp = if isMoE {
            Step3p5MoE(config, layerIndex: layerIndex)
        } else {
            Step3p5MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                limit: config.sharedSwigluLimit(at: layerIndex)
            )
        }
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = attention(inputLayerNorm(input), mask: mask, cache: cache)
        let hidden = input + attentionOutput
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}

private final class Step3p5Backbone: Module {
    private let config: Step3p5Configuration
    private let plan: Step3p5LayerPlan

    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    fileprivate let layers: [Step3p5DecoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: Step3p5Configuration) {
        let layerPlan = Step3p5LayerPlan(config)
        self.config = config
        self.plan = layerPlan
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { layerIndex in
            Step3p5DecoderLayer(config, layerIndex: layerIndex, plan: layerPlan)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)

        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let index = plan.firstFullLayerIndex {
                createAttentionMask(h: hidden, cache: cache?[index])
            } else {
                .none
            }
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let index = plan.firstSlidingLayerIndex {
                createAttentionMask(
                    h: hidden,
                    cache: cache?[index],
                    windowSize: config.slidingWindow
                )
            } else {
                .none
            }

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                mask: layer.isSliding ? slidingMask : fullMask,
                cache: cache?[index]
            )
        }

        return norm(hidden)
    }
}

internal final class Step3p5Model: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let modelType: String
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let config: Step3p5Configuration
    private let plan: Step3p5LayerPlan

    @ModuleInfo(key: "model") private var model: Step3p5Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: Step3p5Configuration) {
        self.config = config
        self.plan = Step3p5LayerPlan(config)
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map {
            Step3p5AttentionLayout(config, layerIndex: $0).keyValueHeads
        }
        _model.wrappedValue = Step3p5Backbone(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hidden = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hidden), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { layerIndex in
            if plan.isSlidingLayer(layerIndex) {
                return RotatingKVCache(maxSize: config.slidingWindow) as KVCache
            }
            if let maxKVSize = parameters?.maxKVSize {
                return RotatingKVCache(
                    maxSize: maxKVSize,
                    keep: GenerationConstants.rotatingCacheKeepTokens
                ) as KVCache
            }
            return KVCacheSimple() as KVCache
        }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert.")
        ]
        let isVanilla = weights.keys.contains { key in
            remappings.contains { source, destination in
                key.contains(source) && !key.contains(destination)
            }
        }

        var sanitized: [String: MLXArray] = [:]
        for (originalKey, originalValue) in weights {
            guard !originalKey.contains(".mtp") else {
                continue
            }
            guard !Self.isExtraLayerKey(originalKey, layerCount: config.hiddenLayers) else {
                continue
            }

            var key = originalKey
            var value = originalValue
            for (source, destination) in remappings where key.contains(source)
                && !key.contains(destination) {
                key = key.replacingOccurrences(of: source, with: destination)
                break
            }
            if isVanilla && key.hasSuffix(".weight") && key.contains("norm") {
                value = value + 1
            }
            sanitized[key] = value
        }

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }
        return sanitized
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }

    private static func isExtraLayerKey(_ key: String, layerCount: Int) -> Bool {
        let parts = key.split(separator: ".")
        guard parts.count > 2, parts[0] == "model", parts[1] == "layers",
              let layerIndex = Int(parts[2]) else {
            return false
        }
        return layerIndex >= layerCount
    }
}

extension Step3p5Model: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.flatMap { layer -> LoRALinearLayers in
            var targets: LoRALinearLayers = [(layer.attention, ["q_proj", "v_proj"])]
            if let mlp = layer.mlp as? Step3p5MLP {
                targets.append((mlp, ["gate_proj", "up_proj"]))
            }
            if let moe = layer.mlp as? Step3p5MoE {
                targets.append((moe.sharedExpert, ["gate_proj", "up_proj"]))
            }
            return targets
        }
    }
}
