import MLX
import MLXNN

internal struct Gemma2AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let keyValueGroups: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float
    internal let attentionLogitSoftCap: Float

    internal init(_ config: Gemma2Configuration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(config.headDimensions > 0, "head_dim must be positive")
        precondition(config.queryPreAttnScalar > 0, "query_pre_attn_scalar must be positive")
        precondition(config.attnLogitSoftcapping > 0, "attn_logit_softcapping must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "num_attention_heads must be divisible by num_key_value_heads"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.keyValueGroups = config.attentionHeads / config.kvHeads
        self.headSize = config.headDimensions
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(config.queryPreAttnScalar).squareRoot()
        self.attentionLogitSoftCap = config.attnLogitSoftcapping
    }
}

private final class Gemma2SelfAttention: Module {
    private let layout: Gemma2AttentionLayout

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    private let rope: RoPE

    init(_ config: Gemma2Configuration) {
        let layout = Gemma2AttentionLayout(config)
        self.layout = layout
        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: false
        )
        self.rope = RoPE(
            dimensions: layout.headSize,
            traditional: config.ropeTraditional,
            base: config.ropeTheta
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        var queries = queryProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            let updated = updateCacheReturningMaterializedKV(
                keys: keys,
                values: values,
                cache: cache
            )
            keys = updated.keys
            values = updated.values
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        queries = queries * layout.attentionScale

        if layout.keyValueGroups > 1 {
            queries = queries.reshaped([
                batchSize,
                layout.keyValueHeads,
                layout.keyValueGroups,
                tokenCount,
                layout.headSize
            ])
            keys = expandedDimensions(keys, axes: [2])
            values = expandedDimensions(values, axes: [2])
        }

        var scores = matmul(queries, keys.swappedAxes(-1, -2))
        scores = tanh(scores / layout.attentionLogitSoftCap) * layout.attentionLogitSoftCap
        if let mask {
            scores = scores + mask
        }

        var attentionOutput = matmul(softmax(scores, axis: -1, precise: true), values)
        if layout.keyValueGroups > 1 {
            attentionOutput = attentionOutput.reshaped([
                batchSize,
                layout.queryHeads,
                tokenCount,
                layout.headSize
            ])
        }

        attentionOutput = attentionOutput
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, -1)

        return outputProjection(attentionOutput)
    }
}

private final class Gemma2FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: Gemma2Configuration) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(gelu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class Gemma2TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma2SelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: Gemma2FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") private var preFeedforwardLayerNorm:
        Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") private var postFeedforwardLayerNorm:
        Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm:
        Gemma.RMSNorm

    init(_ config: Gemma2Configuration) {
        self._selfAttention.wrappedValue = Gemma2SelfAttention(config)
        self._feedForward.wrappedValue = Gemma2FeedForward(config)
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let attentionOutput = selfAttention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        let afterAttention = hiddenStates + postAttentionLayerNorm(attentionOutput)
        let feedForwardOutput = feedForward(preFeedforwardLayerNorm(afterAttention))
        return afterAttention + postFeedforwardLayerNorm(feedForwardOutput)
    }
}

private final class Gemma2Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var tokenEmbeddings: Embedding
    @ModuleInfo var layers: [Gemma2TransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: Gemma.RMSNorm

    private let hiddenScale: Float

    init(_ config: Gemma2Configuration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            Gemma2TransformerBlock(config)
        }
        self._finalNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self.hiddenScale = Float(config.hiddenSize).squareRoot()
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens) * hiddenScale
        let mask: MLXArray? = createAttentionMask(h: tokens, cache: cache)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

internal final class Gemma2Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: Gemma2Backbone
    private let finalLogitSoftCap: Float

    public init(_ config: Gemma2Configuration) {
        precondition(config.finalLogitSoftcapping > 0, "final_logit_softcapping must be positive")

        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = Gemma2Backbone(config)
        self.finalLogitSoftCap = config.finalLogitSoftcapping
    }

    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        softCappedLogits(model.tokenEmbeddings.asLinear(model(tokens, cache: cache)))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(
            logits: softCappedLogits(model.tokenEmbeddings.asLinear(hiddenStates)),
            state: state
        )
    }

    private func softCappedLogits(_ logits: MLXArray) -> MLXArray {
        tanh(logits / finalLogitSoftCap) * finalLogitSoftCap
    }
}

internal struct Gemma2Configuration: Codable, Sendable, Equatable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var attnLogitSoftcapping: Float
    var finalLogitSoftcapping: Float
    var queryPreAttnScalar: Float

    internal init(
        hiddenSize: Int = 2_304,
        hiddenLayers: Int = 26,
        intermediateSize: Int = 9_216,
        attentionHeads: Int = 8,
        headDimensions: Int? = nil,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 256_000,
        kvHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        attnLogitSoftcapping: Float = 50,
        finalLogitSoftcapping: Float = 30,
        queryPreAttnScalar: Float? = nil
    ) {
        let resolvedHeadDimensions = headDimensions ?? max(1, hiddenSize / max(attentionHeads, 1))

        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = resolvedHeadDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.attnLogitSoftcapping = attnLogitSoftcapping
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.queryPreAttnScalar = queryPreAttnScalar ?? Float(resolvedHeadDimensions)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case attnLogitSoftcapping = "attn_logit_softcapping"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case queryPreAttnScalar = "query_pre_attn_scalar"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_304
        let attentionHeads = try container.decodeIfPresent(
            Int.self,
            forKey: .attentionHeads
        ) ?? 8
        let headDimensions = try container.decodeIfPresent(
            Int.self,
            forKey: .headDimensions
        ) ?? max(1, hiddenSize / max(attentionHeads, 1))

        self.init(
            hiddenSize: hiddenSize,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 9_216,
            attentionHeads: attentionHeads,
            headDimensions: headDimensions,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 256_000,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            attnLogitSoftcapping: try container.decodeIfPresent(
                Float.self,
                forKey: .attnLogitSoftcapping
            ) ?? 50,
            finalLogitSoftcapping: try container.decodeIfPresent(
                Float.self,
                forKey: .finalLogitSoftcapping
            ) ?? 30,
            queryPreAttnScalar: try container.decodeIfPresent(
                Float.self,
                forKey: .queryPreAttnScalar
            ) ?? Float(headDimensions)
        )
    }
}

extension Gemma2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
