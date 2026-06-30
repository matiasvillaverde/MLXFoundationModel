import Foundation
import MLX
import MLXNN

internal struct QwenConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var kvChannels: Int
    internal var maxPositionEmbeddings: Int
    internal var layerNormEpsilon: Float
    internal var noBias: Bool
    internal var vocabularySize: Int
    internal var keyValueHeads: Int
    internal var ropeBase: Float
    internal var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "qwen",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvChannels: Int? = nil,
        maxPositionEmbeddings: Int = 8_192,
        layerNormEpsilon: Float = 1e-6,
        noBias: Bool = true,
        vocabularySize: Int,
        keyValueHeads: Int? = nil,
        ropeBase: Float = 10_000,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvChannels = kvChannels ?? hiddenSize / attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.layerNormEpsilon = layerNormEpsilon
        self.noBias = noBias
        self.vocabularySize = vocabularySize
        self.keyValueHeads = keyValueHeads ?? attentionHeads
        self.ropeBase = ropeBase
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvChannels = "kv_channels"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerNormEpsilon = "layer_norm_epsilon"
        case noBias = "no_bias"
        case vocabularySize = "vocab_size"
        case keyValueHeads = "num_key_value_heads"
        case ropeBase = "rotary_emb_base"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen",
            hiddenSize: hiddenSize,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            kvChannels: try container.decodeIfPresent(Int.self, forKey: .kvChannels),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 8_192,
            layerNormEpsilon: try container.decodeIfPresent(
                Float.self,
                forKey: .layerNormEpsilon
            ) ?? 1e-6,
            noBias: try container.decodeIfPresent(Bool.self, forKey: .noBias) ?? true,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads)
                ?? attentionHeads,
            ropeBase: try container.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 10_000,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

internal struct QwenAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let headDim: Int
    internal let keyValueHeads: Int
    internal let projectionSize: Int
    internal let combinedProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: QwenConfiguration) {
        precondition(config.hiddenSize > 0, "Qwen hidden_size must be positive")
        precondition(config.attentionHeads > 0, "Qwen num_attention_heads must be positive")
        precondition(config.keyValueHeads == config.attentionHeads, "Qwen uses full MHA")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "Qwen hidden_size must divide evenly across attention heads"
        )
        precondition(
            config.kvChannels == config.hiddenSize / config.attentionHeads,
            "Qwen kv_channels must match hidden_size / num_attention_heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.headDim = config.kvChannels
        self.keyValueHeads = config.keyValueHeads
        self.projectionSize = config.kvChannels * config.attentionHeads
        self.combinedProjectionSize = 3 * projectionSize
        self.attentionScale = pow(Float(config.hiddenSize / config.attentionHeads), -0.5)
    }
}

private final class QwenAttention: Module {
    let layout: QwenAttentionLayout
    let rope: RoPE

    @ModuleInfo(key: "c_attn") var combinedProjection: Linear
    @ModuleInfo(key: "c_proj") var outputProjection: Linear

    init(_ config: QwenConfiguration) {
        let layout = QwenAttentionLayout(config)
        self.layout = layout
        self.rope = RoPE(dimensions: layout.headDim, traditional: false, base: config.ropeBase)
        _combinedProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.combinedProjectionSize,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            layout.projectionSize,
            layout.hiddenSize,
            bias: !config.noBias
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)
        let splitPoints = [layout.projectionSize, 2 * layout.projectionSize]
        let qkv = split(combinedProjection(hiddenStates), indices: splitPoints, axis: -1)

        var queries = qkv[0]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        var keys = qkv[1]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        let values = qkv[2]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDim)
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
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(output)
    }
}

private final class QwenFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var upProjection: Linear
    @ModuleInfo(key: "w2") var gateProjection: Linear
    @ModuleInfo(key: "c_proj") var downProjection: Linear

    init(_ config: QwenConfiguration) {
        let feedForwardSize = config.intermediateSize / 2
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            feedForwardSize,
            bias: !config.noBias
        )
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            feedForwardSize,
            bias: !config.noBias
        )
        _downProjection.wrappedValue = Linear(
            feedForwardSize,
            config.hiddenSize,
            bias: !config.noBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class QwenTransformerBlock: Module {
    @ModuleInfo(key: "ln_1") var attentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn") var attention: QwenAttention
    @ModuleInfo(key: "ln_2") var feedForwardLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var feedForward: QwenFeedForward

    init(_ config: QwenConfiguration) {
        _attentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _attention.wrappedValue = QwenAttention(config)
        _feedForwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _feedForward.wrappedValue = QwenFeedForward(config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = hiddenStates + attention(
            attentionLayerNorm(hiddenStates),
            mask: mask,
            cache: cache
        )
        return attended + feedForward(feedForwardLayerNorm(attended))
    }
}

private final class QwenBackbone: Module {
    @ModuleInfo(key: "wte") var tokenEmbeddings: Embedding
    @ModuleInfo(key: "h") var layers: [QwenTransformerBlock]
    @ModuleInfo(key: "ln_f") var finalLayerNorm: RMSNorm

    init(_ config: QwenConfiguration) {
        precondition(config.vocabularySize > 0, "Qwen vocab_size must be positive")
        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            QwenTransformerBlock(config)
        }
        _finalLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbeddings(inputs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }

        return finalLayerNorm(hiddenStates)
    }
}

internal final class QwenModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let configuration: QwenConfiguration

    @ModuleInfo(key: "transformer") private var transformer: QwenBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: QwenConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.attentionHeads, count: config.hiddenLayers)
        _transformer.wrappedValue = QwenBackbone(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: !config.noBias
            )
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: transformer(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(
            transformer(input[text: .newAxis].tokens, cache: cache)
        )
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return transformer.tokenEmbeddings.asLinear(hiddenStates)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
        }
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights
    }
}

extension QwenModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        transformer.layers.map { ($0.attention, ["c_attn", "c_proj"]) }
    }
}
