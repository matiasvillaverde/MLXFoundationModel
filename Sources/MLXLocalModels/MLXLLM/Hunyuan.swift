import Foundation
import MLX
import MLXFast
import MLXNN

internal struct HunyuanLayerInt: Codable, Equatable, Sendable {
    private let values: [Int]

    internal init(_ value: Int) {
        self.values = [value]
    }

    internal init(_ values: [Int]) {
        self.values = values
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self.init(value)
        } else {
            self.init(try container.decode([Int].self))
        }
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if values.count == 1 {
            try container.encode(values[0])
        } else {
            try container.encode(values)
        }
    }

    internal func value(at layerIndex: Int) -> Int {
        precondition(!values.isEmpty, "Hunyuan layer value list cannot be empty")
        if values.count == 1 {
            return values[0]
        }
        precondition(
            layerIndex >= 0 && layerIndex < values.count,
            "Hunyuan per-layer value is missing an entry"
        )
        return values[layerIndex]
    }
}

internal struct HunyuanConfiguration: Codable, Equatable, Sendable {
    internal var modelType: String
    internal var vocabularySize: Int
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var kvHeads: Int
    internal var attentionBias: Bool
    internal var moeTopK: HunyuanLayerInt
    internal var numExperts: Int
    internal var numSharedExpert: HunyuanLayerInt
    internal var useMixedMLPMoE: Bool
    internal var useQKNorm: Bool
    internal var rmsNormEps: Float
    internal var ropeTheta: Float
    internal var useCLA: Bool
    internal var claShareFactor: Int
    internal var moeIntermediateSize: HunyuanLayerInt?
    internal var ropeScaling: [String: StringOrNumber]?
    internal var tieWordEmbeddings: Bool
    internal var headDim: Int?
    internal var mlpBias: Bool

    internal var resolvedHeadDim: Int {
        headDim ?? hiddenSize / attentionHeads
    }

    internal var ropeAlpha: Float {
        ropeScaling?["alpha"]?.asFloat() ?? 1
    }

    internal init(
        modelType: String = "hunyuan",
        vocabularySize: Int,
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        attentionBias: Bool = false,
        moeTopK: HunyuanLayerInt = HunyuanLayerInt(1),
        numExperts: Int = 1,
        numSharedExpert: HunyuanLayerInt = HunyuanLayerInt(1),
        useMixedMLPMoE: Bool = false,
        useQKNorm: Bool = true,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000,
        useCLA: Bool = false,
        claShareFactor: Int = 2,
        moeIntermediateSize: HunyuanLayerInt? = nil,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        headDim: Int? = nil,
        mlpBias: Bool = false
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.attentionBias = attentionBias
        self.moeTopK = moeTopK
        self.numExperts = numExperts
        self.numSharedExpert = numSharedExpert
        self.useMixedMLPMoE = useMixedMLPMoE
        self.useQKNorm = useQKNorm
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.useCLA = useCLA
        self.claShareFactor = claShareFactor
        self.moeIntermediateSize = moeIntermediateSize
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.headDim = headDim
        self.mlpBias = mlpBias
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case moeTopK = "moe_topk"
        case numExperts = "num_experts"
        case numSharedExpert = "num_shared_expert"
        case useMixedMLPMoE = "use_mixed_mlp_moe"
        case useQKNorm = "use_qk_norm"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case useCLA = "use_cla"
        case claShareFactor = "cla_share_factor"
        case moeIntermediateSize = "moe_intermediate_size"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case headDim = "head_dim"
        case attentionHeadDim = "attention_head_dim"
        case mlpBias = "mlp_bias"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? container.decodeIfPresent(Int.self, forKey: .attentionHeadDim)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "hunyuan",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: hiddenSize,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            moeTopK: try container.decodeIfPresent(HunyuanLayerInt.self, forKey: .moeTopK)
                ?? HunyuanLayerInt(1),
            numExperts: try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 1,
            numSharedExpert: try container.decodeIfPresent(
                HunyuanLayerInt.self,
                forKey: .numSharedExpert
            ) ?? HunyuanLayerInt(1),
            useMixedMLPMoE: try container.decodeIfPresent(Bool.self, forKey: .useMixedMLPMoE)
                ?? false,
            useQKNorm: try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            useCLA: try container.decodeIfPresent(Bool.self, forKey: .useCLA) ?? false,
            claShareFactor: try container.decodeIfPresent(Int.self, forKey: .claShareFactor)
                ?? 2,
            moeIntermediateSize: try container.decodeIfPresent(
                HunyuanLayerInt.self,
                forKey: .moeIntermediateSize
            ),
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? true,
            headDim: headDim,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: Self.CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(moeTopK, forKey: .moeTopK)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(numSharedExpert, forKey: .numSharedExpert)
        try container.encode(useMixedMLPMoE, forKey: .useMixedMLPMoE)
        try container.encode(useQKNorm, forKey: .useQKNorm)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(useCLA, forKey: .useCLA)
        try container.encode(claShareFactor, forKey: .claShareFactor)
        try container.encodeIfPresent(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encodeIfPresent(headDim, forKey: .headDim)
        try container.encode(mlpBias, forKey: .mlpBias)
    }

    internal func topK(layerIndex: Int) -> Int {
        moeTopK.value(at: layerIndex)
    }

    internal func sharedExpertCount(layerIndex: Int) -> Int {
        numSharedExpert.value(at: layerIndex)
    }

    internal func expertIntermediateSize(layerIndex: Int) -> Int {
        moeIntermediateSize?.value(at: layerIndex) ?? intermediateSize
    }
}

internal struct HunyuanAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headDim: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float
    internal let ropePlan: HunyuanV1DenseRoPEPlan

    internal init(_ config: HunyuanConfiguration) {
        precondition(config.hiddenSize > 0, "Hunyuan hidden_size must be positive")
        precondition(config.attentionHeads > 0, "Hunyuan num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "Hunyuan num_key_value_heads must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "Hunyuan attention heads must divide evenly by KV heads"
        )
        if config.headDim == nil {
            precondition(
                config.hiddenSize.isMultiple(of: config.attentionHeads),
                "Hunyuan hidden_size must divide evenly across attention heads"
            )
        }
        precondition(config.resolvedHeadDim > 0, "Hunyuan head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDim = config.resolvedHeadDim
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.attentionScale = pow(Float(headDim), -0.5)
        self.ropePlan = HunyuanV1DenseRoPEPlan(
            dimensions: headDim,
            base: config.ropeTheta,
            alpha: config.ropeAlpha
        )
    }
}

internal struct HunyuanLayerPlan: Equatable, Sendable {
    internal let hasKeyValueProjection: [Bool]
    internal let usesSparseExperts: [Bool]

    internal init(_ config: HunyuanConfiguration) {
        precondition(config.hiddenLayers > 0, "Hunyuan num_hidden_layers must be positive")
        precondition(config.numExperts > 0, "Hunyuan num_experts must be positive")
        precondition(config.claShareFactor > 0, "Hunyuan cla_share_factor must be positive")

        self.hasKeyValueProjection = (0 ..< config.hiddenLayers).map { layerIndex in
            !config.useCLA || layerIndex.isMultiple(of: config.claShareFactor)
        }
        self.usesSparseExperts = (0 ..< config.hiddenLayers).map { layerIndex in
            config.numExperts > 1 && config.topK(layerIndex: layerIndex) > 0
        }
    }

    internal func projectsKeyValue(layerIndex: Int) -> Bool {
        hasKeyValueProjection[layerIndex]
    }

    internal func usesSparseExperts(layerIndex: Int) -> Bool {
        usesSparseExperts[layerIndex]
    }
}

private final class HunyuanRoPE {
    private let plan: HunyuanV1DenseRoPEPlan
    private let frequencies: MLXArray

    init(plan: HunyuanV1DenseRoPEPlan) {
        self.plan = plan
        self.frequencies = plan.frequencies()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: plan.dimensions,
            traditional: false,
            base: nil,
            scale: 1,
            offset: offset,
            freqs: frequencies
        )
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: plan.dimensions,
            traditional: false,
            base: nil,
            scale: 1,
            offset: offset,
            freqs: frequencies
        )
    }
}

private struct HunyuanKeyValueStates {
    var keys: MLXArray
    var values: MLXArray
}

private final class HunyuanAttention: Module {
    private let layout: HunyuanAttentionLayout
    private let rope: HunyuanRoPE
    private let hasKeyValueProjection: Bool
    private let useQKNorm: Bool

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear?
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear?
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ModuleInfo(key: "query_layernorm") private var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") private var keyNorm: RMSNorm?

    init(_ config: HunyuanConfiguration, hasKeyValueProjection: Bool) {
        let layout = HunyuanAttentionLayout(config)
        self.layout = layout
        self.rope = HunyuanRoPE(plan: layout.ropePlan)
        self.hasKeyValueProjection = hasKeyValueProjection
        self.useQKNorm = config.useQKNorm

        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        if hasKeyValueProjection {
            _keyProjection.wrappedValue = Linear(
                config.hiddenSize,
                layout.keyValueProjectionSize,
                bias: config.attentionBias
            )
            _valueProjection.wrappedValue = Linear(
                config.hiddenSize,
                layout.keyValueProjectionSize,
                bias: config.attentionBias
            )
        }
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: config.attentionBias
        )
        if config.useQKNorm {
            _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
        }
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        sharedKeyValueStates: HunyuanKeyValueStates?
    ) -> (output: MLXArray, keyValueStates: HunyuanKeyValueStates) {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        let projectedKeysValues: HunyuanKeyValueStates
        if let sharedKeyValueStates {
            projectedKeysValues = sharedKeyValueStates
        } else {
            precondition(hasKeyValueProjection, "Hunyuan CLA layer is missing shared KV states")
            guard let keyProjection, let valueProjection else {
                preconditionFailure("Hunyuan layer does not own key/value projections")
            }
            projectedKeysValues = HunyuanKeyValueStates(
                keys: keyProjection(hiddenStates),
                values: valueProjection(hiddenStates)
            )
        }

        var queries = queryProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        var keys = projectedKeysValues.keys
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        let values = projectedKeysValues.values
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(to: queries, cache: cache)
        keys = applyRotaryPosition(to: keys, cache: cache)

        if useQKNorm, let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, layout.queryProjectionSize)

        return (outputProjection(attended), projectedKeysValues)
    }

    private func applyRotaryPosition(to input: MLXArray, cache: KVCache?) -> MLXArray {
        if let batchCache = cache as? BatchPositionedKVCache {
            return rope(input, offset: batchCache.batchOffset)
        }
        return rope(input, offset: cache?.offset ?? 0)
    }
}

private final class HunyuanFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(hiddenSize: Int, intermediateSize: Int, bias: Bool = false) {
        _gateProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _downProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
        _upProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
    }

    convenience init(_ config: HunyuanConfiguration, intermediateSize: Int? = nil) {
        self.init(
            hiddenSize: config.hiddenSize,
            intermediateSize: intermediateSize ?? config.intermediateSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class HunyuanGate: Module {
    @ModuleInfo(key: "wg") private var projection: Linear

    init(hiddenSize: Int, expertCount: Int) {
        _projection.wrappedValue = Linear(hiddenSize, expertCount, bias: false)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        projection(hiddenStates)
    }
}

internal struct HunyuanRoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let topK: Int

    internal init(_ config: HunyuanConfiguration, layerIndex: Int) {
        let topK = config.topK(layerIndex: layerIndex)
        precondition(config.numExperts > 1, "Hunyuan sparse routing requires multiple experts")
        precondition(topK > 0, "Hunyuan moe_topk must be positive")
        precondition(topK <= config.numExperts, "Hunyuan moe_topk cannot exceed num_experts")

        self.expertCount = config.numExperts
        self.topK = topK
    }

    internal func route(logits: MLXArray, outputDType: DType) -> (
        indices: MLXArray, scores: MLXArray
    ) {
        let probabilities = MLX.softmax(logits, axis: -1, precise: true)
        let indices = MLX.argPartition(-probabilities, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let scores = MLX.takeAlong(probabilities, indices, axis: -1).asType(outputDType)
        return (indices, scores)
    }
}

private final class HunyuanSparseFeedForward: Module, UnaryLayer {
    private let routingPlan: HunyuanRoutingPlan

    @ModuleInfo(key: "gate") private var gate: HunyuanGate
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_mlp") private var sharedMLP: HunyuanFeedForward?

    init(_ config: HunyuanConfiguration, layerIndex: Int) {
        let routingPlan = HunyuanRoutingPlan(config, layerIndex: layerIndex)
        self.routingPlan = routingPlan

        _gate.wrappedValue = HunyuanGate(
            hiddenSize: config.hiddenSize,
            expertCount: routingPlan.expertCount
        )
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.expertIntermediateSize(layerIndex: layerIndex),
            numExperts: routingPlan.expertCount
        )
        if config.useMixedMLPMoE {
            let sharedIntermediateSize = config.intermediateSize
                * config.sharedExpertCount(layerIndex: layerIndex)
            _sharedMLP.wrappedValue = HunyuanFeedForward(
                config,
                intermediateSize: sharedIntermediateSize
            )
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let route = routingPlan.route(logits: gate(hiddenStates), outputDType: hiddenStates.dtype)
        let expertOutput = switchMLP(hiddenStates, route.indices)
        var output = (expertOutput * route.scores[.ellipsis, .newAxis])
            .sum(axis: -2)
            .asType(expertOutput.dtype)
        if let sharedMLP {
            output = output + sharedMLP(hiddenStates)
        }
        return output
    }
}

private final class HunyuanDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: HunyuanAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: HunyuanConfiguration, layerIndex: Int, layerPlan: HunyuanLayerPlan) {
        _attention.wrappedValue = HunyuanAttention(
            config,
            hasKeyValueProjection: layerPlan.projectsKeyValue(layerIndex: layerIndex)
        )
        if layerPlan.usesSparseExperts(layerIndex: layerIndex) {
            _feedForward.wrappedValue = HunyuanSparseFeedForward(config, layerIndex: layerIndex)
        } else {
            _feedForward.wrappedValue = HunyuanFeedForward(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        sharedKeyValueStates: HunyuanKeyValueStates?
    ) -> (hiddenStates: MLXArray, keyValueStates: HunyuanKeyValueStates) {
        let attended = attention(
            inputLayerNorm(hiddenStates),
            mask: mask,
            cache: cache,
            sharedKeyValueStates: sharedKeyValueStates
        )
        let afterAttention = hiddenStates + attended.output
        return (
            afterAttention + feedForward(postAttentionLayerNorm(afterAttention)),
            attended.keyValueStates
        )
    }
}

private final class HunyuanBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [HunyuanDecoderLayer]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    private let configuration: HunyuanConfiguration

    init(_ config: HunyuanConfiguration) {
        precondition(config.vocabularySize > 0, "Hunyuan vocab_size must be positive")

        self.configuration = config
        let layerPlan = HunyuanLayerPlan(config)
        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            HunyuanDecoderLayer(config, layerIndex: layerIndex, layerPlan: layerPlan)
        }
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)
        var sharedKeyValueStates: HunyuanKeyValueStates?

        for (layerIndex, layer) in layers.enumerated() {
            if !configuration.useCLA || layerIndex.isMultiple(of: configuration.claShareFactor) {
                sharedKeyValueStates = nil
            }
            let output = layer(
                hiddenStates,
                mask: mask,
                cache: cache?[layerIndex],
                sharedKeyValueStates: sharedKeyValueStates
            )
            hiddenStates = output.hiddenStates
            sharedKeyValueStates = output.keyValueStates
        }

        return finalNorm(hiddenStates)
    }
}

private enum HunyuanExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal struct HunyuanExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: HunyuanConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.numExperts
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights
        guard expertCount > 1 else {
            return packed
        }

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in HunyuanExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let keys = (0 ..< expertCount).map { expertIndex in
                        "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                    }
                    let tensors = keys.compactMap { packed[$0] }
                    guard tensors.count == expertCount else {
                        continue
                    }

                    for key in keys {
                        packed[key] = nil
                    }
                    packed["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        MLX.stacked(tensors)
                }
            }
        }

        return packed
    }
}

private struct HunyuanWeightSanitizer {
    let config: HunyuanConfiguration

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }

        splitLegacyPackedAttention(in: &sanitized)
        splitLegacyPackedFeedForward(in: &sanitized)

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }

        return HunyuanExpertPackingPlan(config).pack(sanitized)
    }

    private func splitLegacyPackedAttention(in weights: inout [String: MLXArray]) {
        let layout = HunyuanAttentionLayout(config)
        let groups = config.attentionHeads / config.kvHeads
        for key in weights.keys.sorted() where key.hasSuffix(".self_attn.qkv_proj.weight") {
            let prefix = String(key.dropLast(".qkv_proj.weight".count))
            guard let packed = weights.removeValue(forKey: key) else {
                continue
            }
            let shaped = packed.reshaped(config.kvHeads, groups + 2, layout.headDim, -1)
            let parts = split(shaped, indices: [groups, groups + 1], axis: 1)
            weights["\(prefix).q_proj.weight"] = parts[0].flattened(start: 0, end: 2)
            weights["\(prefix).k_proj.weight"] = parts[1].flattened(start: 0, end: 2)
            weights["\(prefix).v_proj.weight"] = parts[2].flattened(start: 0, end: 2)
        }
    }

    private func splitLegacyPackedFeedForward(in weights: inout [String: MLXArray]) {
        for key in weights.keys.sorted() where key.hasSuffix(".mlp.gate_and_up_proj.weight") {
            let prefix = String(key.dropLast(".gate_and_up_proj.weight".count))
            guard let packed = weights.removeValue(forKey: key) else {
                continue
            }
            let parts = split(packed, parts: 2, axis: 0)
            weights["\(prefix).up_proj.weight"] = parts[0]
            weights["\(prefix).gate_proj.weight"] = parts[1]
        }
    }
}

internal final class HunyuanModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let configuration: HunyuanConfiguration

    @ModuleInfo(key: "model") private var model: HunyuanBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: HunyuanConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        _model.wrappedValue = HunyuanBackbone(configuration)

        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        HunyuanWeightSanitizer(config: configuration).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

extension HunyuanModel: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
