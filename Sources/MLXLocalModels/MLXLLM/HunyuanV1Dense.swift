import Foundation
import MLX
import MLXFast
import MLXNN

internal struct HunyuanV1DenseRoPEPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let base: Float
    internal let alpha: Float
    internal let adjustedBase: Float

    internal init(dimensions: Int, base: Float, alpha: Float = 1) {
        precondition(dimensions > 2, "Hunyuan head_dim must be greater than 2 for dynamic RoPE")
        precondition(dimensions.isMultiple(of: 2), "Hunyuan head_dim must be even")
        precondition(base > 0, "Hunyuan rope_theta must be positive")
        precondition(alpha > 0, "Hunyuan dynamic RoPE alpha must be positive")

        self.dimensions = dimensions
        self.base = base
        self.alpha = alpha
        self.adjustedBase = base * pow(alpha, Float(dimensions) / Float(dimensions - 2))
    }

    internal func frequencies() -> MLXArray {
        let exponents = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
            / Float(dimensions)
        return pow(adjustedBase, exponents)
    }
}

private final class HunyuanV1DenseRoPE {
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

internal struct HunyuanV1DenseConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var vocabularySize: Int
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var kvHeads: Int
    internal var rmsNormEps: Float
    internal var ropeTheta: Float
    internal var maxPositionEmbeddings: Int
    internal var attentionBias: Bool
    internal var useQKNorm: Bool
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
        modelType: String = "hunyuan_v1_dense",
        vocabularySize: Int,
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000,
        maxPositionEmbeddings: Int = 32_768,
        attentionBias: Bool = false,
        useQKNorm: Bool = true,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
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
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.useQKNorm = useQKNorm
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
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case useQKNorm = "use_qk_norm"
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
                ?? "hunyuan_v1_dense",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: hiddenSize,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            kvHeads: try container.decode(Int.self, forKey: .kvHeads),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 32_768,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            useQKNorm: try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? false,
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
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(useQKNorm, forKey: .useQKNorm)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encodeIfPresent(headDim, forKey: .headDim)
        try container.encode(mlpBias, forKey: .mlpBias)
    }
}

internal struct HunyuanV1DenseAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headDim: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float
    internal let ropePlan: HunyuanV1DenseRoPEPlan

    internal init(_ config: HunyuanV1DenseConfiguration) {
        precondition(config.hiddenSize > 0, "Hunyuan hidden_size must be positive")
        precondition(config.attentionHeads > 0, "Hunyuan num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "Hunyuan num_key_value_heads must be positive")
        if config.headDim == nil {
            precondition(
                config.hiddenSize.isMultiple(of: config.attentionHeads),
                "Hunyuan hidden_size must be divisible by num_attention_heads"
            )
        }
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "Hunyuan attention heads must be divisible by KV heads"
        )
        precondition(config.resolvedHeadDim > 0, "Hunyuan head_dim must be positive")

        let headDim = config.resolvedHeadDim
        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDim = headDim
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

private final class HunyuanV1DenseAttention: Module {
    private let layout: HunyuanV1DenseAttentionLayout
    private let rope: HunyuanV1DenseRoPE
    private let useQKNorm: Bool

    @ModuleInfo(key: "q_proj") fileprivate var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") fileprivate var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ModuleInfo(key: "query_layernorm") private var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") private var keyNorm: RMSNorm?

    init(_ config: HunyuanV1DenseConfiguration) {
        let layout = HunyuanV1DenseAttentionLayout(config)
        self.layout = layout
        self.rope = HunyuanV1DenseRoPE(plan: layout.ropePlan)
        self.useQKNorm = config.useQKNorm

        _queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        _keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
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
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        var queries = queryProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(to: queries, cache: cache)
        keys = applyRotaryPosition(to: keys, cache: cache)

        if useQKNorm, let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, layout.queryProjectionSize)

        return outputProjection(output)
    }

    private func applyRotaryPosition(to input: MLXArray, cache: KVCache?) -> MLXArray {
        if let batchCache = cache as? BatchPositionedKVCache {
            return rope(input, offset: batchCache.batchOffset)
        }
        return rope(input, offset: cache?.offset ?? 0)
    }
}

private final class HunyuanV1DenseFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: HunyuanV1DenseConfiguration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class HunyuanV1DenseDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: HunyuanV1DenseAttention
    @ModuleInfo(key: "mlp") private var feedForward: HunyuanV1DenseFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: HunyuanV1DenseConfiguration) {
        _selfAttention.wrappedValue = HunyuanV1DenseAttention(config)
        _feedForward.wrappedValue = HunyuanV1DenseFeedForward(config)
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
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = hiddenStates
            + selfAttention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class HunyuanV1DenseBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [HunyuanV1DenseDecoderLayer]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: HunyuanV1DenseConfiguration) {
        precondition(config.vocabularySize > 0, "Hunyuan vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "Hunyuan num_hidden_layers must be positive")

        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            HunyuanV1DenseDecoderLayer(config)
        }
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

internal final class HunyuanV1DenseModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let configuration: HunyuanV1DenseConfiguration

    @ModuleInfo(key: "model") private var backbone: HunyuanV1DenseBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: HunyuanV1DenseConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        _backbone.wrappedValue = HunyuanV1DenseBackbone(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: backbone(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(
            backbone(input[text: .newAxis].tokens, cache: cache)
        )
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["lm_head.scales"] = nil
            weights["lm_head.biases"] = nil
        }
        return weights
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return backbone.tokenEmbeddings.asLinear(hiddenStates)
    }
}

extension HunyuanV1DenseModel: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        backbone.layers.map { layer in
            (layer.selfAttention, ["q_proj", "v_proj"])
        }
    }
}
