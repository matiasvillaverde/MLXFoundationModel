import Foundation
import MLX
import MLXFast
import MLXNN

internal struct GLMConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var rmsNormEps: Float
    internal var vocabularySize: Int
    internal var headDim: Int?
    internal var kvHeads: Int
    internal var maxPositionEmbeddings: Int?
    internal var attentionBias: Bool
    internal var ropeTheta: Float
    internal var tieWordEmbeddings: Bool

    internal var resolvedHeadDim: Int {
        headDim ?? hiddenSize / attentionHeads
    }

    internal init(
        modelType: String = "glm",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int,
        headDim: Int? = nil,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        attentionBias: Bool = false,
        ropeTheta: Float = 10_000,
        tieWordEmbeddings: Bool = true
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDim = headDim
        self.kvHeads = kvHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDim = "head_dim"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "glm",
            hiddenSize: hiddenSize,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? true
        )
    }
}

internal struct GLMAttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headDim: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: GLMConfiguration) {
        precondition(config.hiddenSize > 0, "GLM hidden_size must be positive")
        precondition(config.attentionHeads > 0, "GLM num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "GLM num_key_value_heads must be positive")
        if config.headDim == nil {
            precondition(
                config.hiddenSize.isMultiple(of: config.attentionHeads),
                "GLM hidden_size must be divisible by num_attention_heads"
            )
        }
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "GLM attention heads must be divisible by KV heads"
        )
        precondition(config.resolvedHeadDim > 0, "GLM head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDim = config.resolvedHeadDim
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.attentionScale = pow(Float(headDim), -0.5)
    }
}

private final class GLMAttention: Module {
    private let layout: GLMAttentionLayout
    private let rope: RoPE

    @ModuleInfo(key: "q_proj") fileprivate var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") fileprivate var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: GLMConfiguration) {
        let layout = GLMAttentionLayout(config)
        self.layout = layout
        self.rope = RoPE(
            dimensions: layout.headDim,
            traditional: true,
            base: config.ropeTheta
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
            bias: false
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

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
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
}

private final class GLMFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") private var gateUpProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: GLMConfiguration) {
        _gateUpProjection.wrappedValue = Linear(
            config.hiddenSize,
            2 * config.intermediateSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let parts = split(gateUpProjection(hiddenStates), parts: 2, axis: -1)
        return downProjection(silu(parts[0]) * parts[1])
    }
}

private final class GLMDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: GLMAttention
    @ModuleInfo(key: "mlp") private var feedForward: GLMFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: GLMConfiguration) {
        _selfAttention.wrappedValue = GLMAttention(config)
        _feedForward.wrappedValue = GLMFeedForward(config)
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

private final class GLMBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [GLMDecoderLayer]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: GLMConfiguration) {
        precondition(config.vocabularySize > 0, "GLM vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "GLM num_hidden_layers must be positive")

        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            GLMDecoderLayer(config)
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

internal final class GLMModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let configuration: GLMConfiguration

    @ModuleInfo(key: "model") private var backbone: GLMBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: GLMConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        _backbone.wrappedValue = GLMBackbone(configuration)
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

extension GLMModel: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        backbone.layers.map { layer in
            (layer.selfAttention, ["q_proj", "v_proj"])
        }
    }
}
