import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

internal struct StableLMConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float
    var useQKVBias: Bool
    var partialRotaryFactor: Float
    var layerNormEps: Float
    var useParallelResidual: Bool
    var qkLayerNorm: Bool
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "stablelm",
        vocabularySize: Int = 100_352,
        hiddenSize: Int = 2_048,
        hiddenLayers: Int = 24,
        intermediateSize: Int = 5_632,
        attentionHeads: Int = 32,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        ropeTheta: Float = 10_000,
        useQKVBias: Bool = false,
        partialRotaryFactor: Float = 0.25,
        layerNormEps: Float = 1e-5,
        useParallelResidual: Bool = false,
        qkLayerNorm: Bool = false,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.useQKVBias = useQKVBias
        self.partialRotaryFactor = partialRotaryFactor
        self.layerNormEps = layerNormEps
        self.useParallelResidual = useParallelResidual
        self.qkLayerNorm = qkLayerNorm
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case useQKVBias = "use_qkv_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case layerNormEps = "layer_norm_eps"
        case layerNormEpsilon = "layer_norm_epsilon"
        case useParallelResidual = "use_parallel_residual"
        case qkLayerNorm = "qk_layernorm"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? 32
        let layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
            ?? container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon)
            ?? 1e-5

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "stablelm",
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 100_352,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? 5_632,
            attentionHeads: attentionHeads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000,
            useQKVBias: try container.decodeIfPresent(Bool.self, forKey: .useQKVBias)
                ?? false,
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 0.25,
            layerNormEps: layerNormEps,
            useParallelResidual: try container.decodeIfPresent(
                Bool.self,
                forKey: .useParallelResidual
            ) ?? false,
            qkLayerNorm: try container.decodeIfPresent(Bool.self, forKey: .qkLayerNorm)
                ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encodeIfPresent(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(useQKVBias, forKey: .useQKVBias)
        try container.encode(partialRotaryFactor, forKey: .partialRotaryFactor)
        try container.encode(layerNormEps, forKey: .layerNormEps)
        try container.encode(useParallelResidual, forKey: .useParallelResidual)
        try container.encode(qkLayerNorm, forKey: .qkLayerNorm)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }
}

// MARK: - Plans

internal struct StableLMAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let rotaryDimensions: Int
    internal let attentionScale: Float
    internal let usesPerHeadLayerNorm: Bool

    internal init(_ config: StableLMConfiguration) {
        precondition(config.hiddenSize > 0, "StableLM hidden size must be positive")
        precondition(config.attentionHeads > 0, "StableLM attention heads must be positive")
        precondition(config.kvHeads > 0, "StableLM KV heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "StableLM hidden size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "StableLM attention heads must group KV heads"
        )
        precondition(
            config.partialRotaryFactor > 0,
            "StableLM partial rotary factor must be positive"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.rotaryDimensions = max(
            1,
            Int(Float(headDimensions) * config.partialRotaryFactor)
        )
        self.attentionScale = pow(Float(headDimensions), -0.5)
        self.usesPerHeadLayerNorm = config.qkLayerNorm
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
}

// MARK: - Layers

internal final class StableLMLayerNormPerHead: Module {
    @ModuleInfo(key: "norms") var norms: [LayerNorm]

    let headDimensions: Int
    let headCount: Int
    let eps: Float

    init(headDimensions: Int, headCount: Int, eps: Float) {
        self.headDimensions = headDimensions
        self.headCount = headCount
        self.eps = eps
        _norms.wrappedValue = (0 ..< headCount).map { _ in
            LayerNorm(dimensions: headDimensions, eps: eps, affine: true, bias: false)
        }
    }

    func callAsFunction(_ states: MLXArray) -> MLXArray {
        let weight = MLX.stacked(norms.map { norm in
            norm.weight ?? MLXArray.ones([headDimensions])
        })
        return weight * MLXFast.layerNorm(states, weight: nil, bias: nil, eps: eps)
    }
}

internal final class StableLMAttention: Module {
    let layout: StableLMAttentionLayout

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_layernorm") var queryLayerNorm: StableLMLayerNormPerHead?
    @ModuleInfo(key: "k_layernorm") var keyLayerNorm: StableLMLayerNormPerHead?

    let rope: RoPE

    init(_ config: StableLMConfiguration) {
        self.layout = StableLMAttentionLayout(config)

        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: config.useQKVBias
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.useQKVBias
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.useQKVBias
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: false
        )

        if layout.usesPerHeadLayerNorm {
            _queryLayerNorm.wrappedValue = StableLMLayerNormPerHead(
                headDimensions: layout.headDimensions,
                headCount: layout.attentionHeads,
                eps: config.layerNormEps
            )
            _keyLayerNorm.wrappedValue = StableLMLayerNormPerHead(
                headDimensions: layout.headDimensions,
                headCount: layout.keyValueHeads,
                eps: config.layerNormEps
            )
        }

        self.rope = RoPE(
            dimensions: layout.rotaryDimensions,
            traditional: false,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        var queries = queryProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
        let values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        if let queryLayerNorm, let keyLayerNorm {
            queries = queryLayerNorm(queries)
            keys = keyLayerNorm(keys)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let attentionOutput = attentionWithCacheUpdate(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .asType(values.dtype)
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(attentionOutput)
    }
}

internal final class StableLMFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear

    init(_ config: StableLMConfiguration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

internal final class StableLMDecoderLayer: Module {
    let useParallelResidual: Bool

    @ModuleInfo(key: "self_attn") var attention: StableLMAttention
    @ModuleInfo(key: "mlp") var feedForward: StableLMFeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm?

    init(_ config: StableLMConfiguration) {
        self.useParallelResidual = config.useParallelResidual

        _attention.wrappedValue = StableLMAttention(config)
        _feedForward.wrappedValue = StableLMFeedForward(config)
        _inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        if !config.useParallelResidual {
            _postAttentionLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps
            )
        }
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normalizedStates = inputLayerNorm(hiddenStates)
        let attentionOutput = attention(normalizedStates, mask: mask, cache: cache)

        if useParallelResidual {
            return hiddenStates + attentionOutput + feedForward(normalizedStates)
        }

        let afterAttention = hiddenStates + attentionOutput
        guard let postAttentionLayerNorm else {
            preconditionFailure("StableLM sequential residual path requires post-attention norm")
        }
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

internal final class StableLMBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [StableLMDecoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: StableLMConfiguration) {
        precondition(config.vocabularySize > 0, "StableLM vocabulary size must be positive")
        precondition(config.hiddenLayers > 0, "StableLM must have at least one layer")

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            StableLMDecoderLayer(config)
        }
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embedTokens(inputs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }

        return norm(hiddenStates)
    }
}

internal final class StableLMModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: StableLMConfiguration

    @ModuleInfo(key: "model") var model: StableLMBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: StableLMConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _model.wrappedValue = StableLMBackbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
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

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
        if configuration.tieWordEmbeddings {
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
}

extension StableLMModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
