import Foundation
import MLX
import MLXFast
import MLXNN

internal struct TeleChat3Configuration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var intermediateSize: Int
    internal var maxPositionEmbeddings: Int
    internal var attentionHeads: Int
    internal var hiddenLayers: Int
    internal var keyValueHeads: Int
    internal var rmsNormEps: Float
    internal var vocabularySize: Int
    internal var ropeTheta: Float
    internal var mlpBias: Bool
    internal var attentionBias: Bool
    internal var headDim: Int?
    internal var ropeScaling: [String: StringOrNumber]?
    internal var tieWordEmbeddings: Bool

    internal var resolvedHeadDim: Int {
        headDim ?? hiddenSize / attentionHeads
    }

    internal init(
        modelType: String = "telechat3",
        hiddenSize: Int,
        intermediateSize: Int,
        maxPositionEmbeddings: Int = 32_768,
        attentionHeads: Int,
        hiddenLayers: Int,
        keyValueHeads: Int? = nil,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int,
        ropeTheta: Float = 1_000_000,
        mlpBias: Bool = false,
        attentionBias: Bool = false,
        headDim: Int? = nil,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionHeads = attentionHeads
        self.hiddenLayers = hiddenLayers
        self.keyValueHeads = keyValueHeads ?? attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.mlpBias = mlpBias
        self.attentionBias = attentionBias
        self.headDim = headDim
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case keyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case mlpBias = "mlp_bias"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "telechat3",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 32_768,
            attentionHeads: attentionHeads,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 1_000_000,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

internal struct TeleChat3AttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headDim: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: TeleChat3Configuration) {
        precondition(config.hiddenSize > 0, "TeleChat3 hidden_size must be positive")
        precondition(config.attentionHeads > 0, "TeleChat3 num_attention_heads must be positive")
        precondition(config.keyValueHeads > 0, "TeleChat3 num_key_value_heads must be positive")
        if config.headDim == nil {
            precondition(
                config.hiddenSize.isMultiple(of: config.attentionHeads),
                "TeleChat3 hidden_size must divide evenly across attention heads"
            )
        }
        precondition(
            config.attentionHeads.isMultiple(of: config.keyValueHeads),
            "TeleChat3 attention heads must be divisible by KV heads"
        )
        precondition(config.resolvedHeadDim > 0, "TeleChat3 head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.headDim = config.resolvedHeadDim
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.attentionScale = 1 / Float(headDim).squareRoot()
    }
}

private final class TeleChat3Attention: Module {
    private let layout: TeleChat3AttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") fileprivate var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") fileprivate var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: TeleChat3Configuration) {
        let layout = TeleChat3AttentionLayout(config)
        self.layout = layout
        self.rope = initializeRope(
            dims: layout.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
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

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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
}

private final class TeleChat3FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: TeleChat3Configuration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class TeleChat3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: TeleChat3Attention
    @ModuleInfo(key: "mlp") private var feedForward: TeleChat3FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: TeleChat3Configuration) {
        _selfAttention.wrappedValue = TeleChat3Attention(config)
        _feedForward.wrappedValue = TeleChat3FeedForward(config)
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
        let attentionOutput = selfAttention(
            inputLayerNorm(hiddenStates),
            mask: mask,
            cache: cache
        )
        let afterAttention = hiddenStates + attentionOutput
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class TeleChat3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [TeleChat3DecoderLayer]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: TeleChat3Configuration) {
        precondition(config.vocabularySize > 0, "TeleChat3 vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "TeleChat3 num_hidden_layers must be positive")

        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            TeleChat3DecoderLayer(config)
        }
        _finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
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

internal final class TeleChat3Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let configuration: TeleChat3Configuration

    @ModuleInfo(key: "model") private var backbone: TeleChat3Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: TeleChat3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.keyValueHeads, count: configuration.hiddenLayers)
        _backbone.wrappedValue = TeleChat3Backbone(configuration)
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

extension TeleChat3Model: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        backbone.layers.map { layer in
            (layer.selfAttention, ["q_proj", "v_proj"])
        }
    }
}
