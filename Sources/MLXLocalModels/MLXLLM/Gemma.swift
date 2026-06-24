import MLX
import MLXFast
import MLXNN

internal enum Gemma {
    internal final class RMSNorm: Module, UnaryLayer {
        private let weight: MLXArray
        private let eps: Float

        internal init(dimensions: Int, eps: Float = 1e-5) {
            self.weight = MLXArray.ones([dimensions])
            self.eps = eps
        }

        internal func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(hiddenStates, weight: 1 + weight, eps: eps)
        }
    }

    internal static func clipResidual(_ lhs: MLXArray, _ rhs: MLXArray) -> MLXArray {
        guard lhs.dtype == .float16 else {
            return lhs + rhs
        }

        let float16Max: Float = 65_504
        let sum = lhs.asType(.float32) + rhs.asType(.float32)
        return clip(sum, min: MLXArray(-float16Max), max: MLXArray(float16Max))
            .asType(.float16)
    }
}

internal struct GemmaAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: GemmaConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(config.headDimensions > 0, "head_dim must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "num_attention_heads must be divisible by num_key_value_heads"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = config.headDimensions
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

private final class GemmaSelfAttention: Module {
    private let layout: GemmaAttentionLayout

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    private let rope: RoPE

    init(_ config: GemmaConfiguration) {
        let layout = GemmaAttentionLayout(config)
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

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        var queries = queryProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(hiddenStates)
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
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(attentionOutput)
    }
}

private final class GemmaFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: GemmaConfiguration) {
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

private final class GemmaTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: GemmaSelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: GemmaFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm:
        Gemma.RMSNorm

    init(_ config: GemmaConfiguration) {
        self._selfAttention.wrappedValue = GemmaSelfAttention(config)
        self._feedForward.wrappedValue = GemmaFeedForward(config)
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
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

private final class GemmaBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var tokenEmbeddings: Embedding
    @ModuleInfo var layers: [GemmaTransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: Gemma.RMSNorm

    private let hiddenScale: Float

    init(_ config: GemmaConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            GemmaTransformerBlock(config)
        }
        self._finalNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self.hiddenScale = Float(config.hiddenSize).squareRoot()
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens) * hiddenScale
        let mask = createAttentionMask(h: tokens, cache: cache)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

internal final class GemmaModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    internal let modelType: String
    fileprivate let model: GemmaBackbone

    public init(_ config: GemmaConfiguration) {
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = GemmaBackbone(config)
    }

    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.tokenEmbeddings.asLinear(model(tokens, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: model.tokenEmbeddings.asLinear(hiddenStates), state: state)
    }
}

internal struct GemmaConfiguration: Codable, Sendable, Equatable {
    var modelType: String
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

    internal init(
        modelType: String = "gemma",
        hiddenSize: Int = 2_048,
        hiddenLayers: Int = 18,
        intermediateSize: Int = 16_384,
        attentionHeads: Int = 8,
        headDimensions: Int? = nil,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 256_000,
        kvHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions ?? max(1, hiddenSize / max(attentionHeads, 1))
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
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
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048
        let attentionHeads = try container.decodeIfPresent(
            Int.self,
            forKey: .attentionHeads
        ) ?? 8

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma",
            hiddenSize: hiddenSize,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 18,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 16_384,
            attentionHeads: attentionHeads,
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
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
            ) ?? false
        )
    }
}

extension GemmaModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
