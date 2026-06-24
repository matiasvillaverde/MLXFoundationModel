import Foundation
import MLX
import MLXNN

internal struct Ernie45AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: Ernie45Configuration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.numAttentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.numKeyValueHeads > 0, "num_key_value_heads must be positive")

        let headSize = Self.resolveHeadSize(config)
        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.numAttentionHeads
        self.keyValueHeads = config.numKeyValueHeads
        self.headSize = headSize
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }

    private static func resolveHeadSize(_ config: Ernie45Configuration) -> Int {
        if let headDim = config.headDim {
            precondition(headDim > 0, "head_dim must be positive")
            return headDim
        }

        precondition(
            config.hiddenSize % config.numAttentionHeads == 0,
            "hidden_size must be divisible by num_attention_heads when head_dim is absent"
        )
        return config.hiddenSize / config.numAttentionHeads
    }
}

private final class Ernie45Attention: Module {
    private let layout: Ernie45AttentionLayout
    private let rope: RoPE

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: Ernie45Configuration) {
        let layout = Ernie45AttentionLayout(config)
        self.layout = layout
        self.rope = RoPE(
            dimensions: layout.headSize,
            traditional: true,
            base: config.ropeTheta
        )

        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.useBias
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.useBias
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.useBias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: config.useBias
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)

        var queries = queryProjection(input)
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        return outputProjection(
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.attentionScale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, layout.queryProjectionSize)
        )
    }
}

private final class Ernie45FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: Ernie45Configuration) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.useBias
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.useBias
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.useBias
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class Ernie45Block: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: Ernie45Attention
    @ModuleInfo(key: "mlp") private var feedForward: Ernie45FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: Ernie45Configuration) {
        self._attention.wrappedValue = Ernie45Attention(config)
        self._feedForward.wrappedValue = Ernie45FeedForward(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = attention(
            inputLayerNorm(input),
            mask: mask,
            cache: cache
        )
        let afterAttention = input + attentionOutput
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class Ernie45Backbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embeddings: Embedding
    @ModuleInfo fileprivate var layers: [Ernie45Block]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: Ernie45Configuration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            Ernie45Block(config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embeddings(tokens)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                mask: mask,
                cache: cache?[index]
            )
        }

        return norm(hiddenStates)
    }
}

internal final class Ernie45Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: Ernie45Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public init(_ config: Ernie45Configuration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads,
            count: config.numHiddenLayers
        )
        self._model.wrappedValue = Ernie45Backbone(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
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
        greedyTokenOutput(
            logits: logits(
                from: lastTokenHiddenState(
                    model(input[text: .newAxis].tokens, cache: cache)
                )
            ),
            state: state
        )
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) }
            ?? model.embeddings.asLinear(hiddenStates)
    }
}

internal struct Ernie45Configuration: Codable, Sendable, Equatable {
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int?
    var numHiddenLayers: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var useBias: Bool
    var tieWordEmbeddings: Bool

    internal init(
        hiddenSize: Int = 1_024,
        intermediateSize: Int = 3_072,
        maxPositionEmbeddings: Int = 131_072,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 2,
        headDim: Int? = nil,
        numHiddenLayers: Int = 18,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int = 103_424,
        ropeTheta: Float = 500_000,
        useBias: Bool = false,
        tieWordEmbeddings: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.numHiddenLayers = numHiddenLayers
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.useBias = useBias
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numHiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case useBias = "use_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        self.init(
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1_024,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 3_072,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            numAttentionHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .numAttentionHeads
            ) ?? 16,
            numKeyValueHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .numKeyValueHeads
            ) ?? 2,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            numHiddenLayers: try container.decodeIfPresent(
                Int.self,
                forKey: .numHiddenLayers
            ) ?? 18,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 103_424,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 500_000,
            useBias: try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true
        )
    }
}

extension Ernie45Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
