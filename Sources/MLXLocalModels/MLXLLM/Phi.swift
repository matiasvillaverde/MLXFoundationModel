import MLX
import MLXFast
import MLXNN

internal struct PhiAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let rotaryDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: PhiConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "hidden_size must be divisible by num_attention_heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        precondition(config.partialRotaryFactor >= 0, "partial_rotary_factor cannot be negative")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = config.hiddenSize / config.attentionHeads
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.rotaryDimensions = min(
            headSize,
            Int(Float(headSize) * config.partialRotaryFactor)
        )
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

private final class PhiSelfAttention: Module {
    private let layout: PhiAttentionLayout

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "dense") private var outputProjection: Linear

    private let rope: RoPE

    init(_ config: PhiConfiguration) {
        let layout = PhiAttentionLayout(config)
        self.layout = layout
        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: true
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: true
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: true
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: true
        )
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
        var keys = keyProjection(hiddenStates)
        var values = valueProjection(hiddenStates)

        queries = queries
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        keys = keys
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        values = values
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let attentionOutput = attentionWithCacheUpdate(
            queries: queries.asType(.float32),
            keys: keys,
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

private final class PhiFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") private var upProjection: Linear
    @ModuleInfo(key: "fc2") private var downProjection: Linear

    private let activation = GELU(approximation: .precise)

    init(_ config: PhiConfiguration) {
        self._upProjection.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        self._downProjection.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(activation(upProjection(hiddenStates)))
    }
}

private final class PhiTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: PhiSelfAttention
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "mlp") private var feedForward: PhiFeedForward

    init(_ config: PhiConfiguration) {
        self._selfAttention.wrappedValue = PhiSelfAttention(config)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        self._feedForward.wrappedValue = PhiFeedForward(config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normalizedStates = inputLayerNorm(hiddenStates)
        return hiddenStates
            + selfAttention(normalizedStates, mask: mask, cache: cache)
            + feedForward(normalizedStates)
    }
}

private final class PhiBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [PhiTransformerBlock]
    @ModuleInfo(key: "final_layernorm") private var finalLayerNorm: LayerNorm

    init(_ config: PhiConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            PhiTransformerBlock(config)
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(
        _ tokens: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var hiddenStates = embedTokens(tokens)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalLayerNorm(hiddenStates)
    }
}

internal final class PhiModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: PhiBackbone

    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    public init(_ config: PhiConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = PhiBackbone(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: true)
    }

    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        let mask = createAttentionMask(h: tokens, cache: cache)
        return lmHead(model(tokens, mask: mask, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let tokens = input[text: .newAxis].tokens
        let mask = createAttentionMask(h: tokens, cache: cache)
        let hiddenStates = model(tokens, mask: mask, cache: cache)
        return greedyTokenOutput(logits: lmHead(lastTokenHiddenState(hiddenStates)), state: state)
    }
}

internal struct PhiConfiguration: Codable, Sendable, Equatable {
    var maxPositionalEmbeddings: Int
    var vocabularySize: Int
    var hiddenSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var partialRotaryFactor: Float
    var intermediateSize: Int
    var layerNormEps: Float
    var ropeTheta: Float

    internal init(
        maxPositionalEmbeddings: Int = 2_048,
        vocabularySize: Int = 51_200,
        hiddenSize: Int = 2_560,
        attentionHeads: Int = 32,
        hiddenLayers: Int = 32,
        kvHeads: Int? = nil,
        partialRotaryFactor: Float = 0.4,
        intermediateSize: Int = 10_240,
        layerNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000
    ) {
        self.maxPositionalEmbeddings = maxPositionalEmbeddings
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.attentionHeads = attentionHeads
        self.hiddenLayers = hiddenLayers
        self.kvHeads = kvHeads ?? attentionHeads
        self.partialRotaryFactor = partialRotaryFactor
        self.intermediateSize = intermediateSize
        self.layerNormEps = layerNormEps
        self.ropeTheta = ropeTheta
    }

    enum CodingKeys: String, CodingKey {
        case maxPositionalEmbeddings = "max_position_embeddings"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case partialRotaryFactor = "partial_rotary_factor"
        case intermediateSize = "intermediate_size"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        self.init(
            maxPositionalEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionalEmbeddings
            ) ?? 2_048,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 51_200,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_560,
            attentionHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .attentionHeads
            ) ?? 32,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 0.4,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 10_240,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        )
    }
}

extension PhiModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
