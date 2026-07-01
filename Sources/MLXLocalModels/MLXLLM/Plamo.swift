import Foundation
import MLX
import MLXFast
import MLXNN

internal struct PlamoConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var rmsNormEps: Float
    internal var vocabularySize: Int
    internal var sharedHeadGroupSize: Int
    internal var ropeTheta: Float
    internal var ropeTraditional: Bool
    internal var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "plamo",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int,
        sharedHeadGroupSize: Int = 8,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.sharedHeadGroupSize = sharedHeadGroupSize
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    internal var headDimensions: Int {
        hiddenSize / attentionHeads
    }

    internal var keyValueHeads: Int {
        attentionHeads / sharedHeadGroupSize
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case sharedHeadGroupSize = "n_shared_head"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "plamo",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-6,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            sharedHeadGroupSize: try container.decodeIfPresent(
                Int.self,
                forKey: .sharedHeadGroupSize
            ) ?? 8,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

internal struct PlamoAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let sharedHeadGroupSize: Int
    internal let headDimensions: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: PlamoConfiguration) {
        precondition(config.hiddenSize > 0, "Plamo hidden_size must be positive")
        precondition(config.attentionHeads > 0, "Plamo num_attention_heads must be positive")
        precondition(config.sharedHeadGroupSize > 0, "Plamo n_shared_head must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "Plamo hidden_size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.sharedHeadGroupSize),
            "Plamo attention heads must divide evenly across shared-head groups"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.sharedHeadGroupSize = config.sharedHeadGroupSize
        self.headDimensions = config.headDimensions
        self.queryProjectionSize = queryHeads * headDimensions
        self.keyValueProjectionSize = keyValueHeads * headDimensions
        self.attentionScale = 1 / Float(headDimensions).squareRoot()
    }
}

private final class PlamoAttention: Module {
    let layout: PlamoAttentionLayout
    let rope: RoPE

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: PlamoConfiguration) {
        let layout = PlamoAttentionLayout(config)
        self.layout = layout
        self.rope = RoPE(
            dimensions: layout.headDimensions,
            traditional: config.ropeTraditional,
            base: config.ropeTheta
        )
        _queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        _keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
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
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            let cached = updateCacheReturningMaterializedKV(keys: keys, values: values, cache: cache)
            keys = cached.keys
            values = cached.values
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: expandedSharedHeads(keys),
            values: expandedSharedHeads(values),
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(attended)
    }

    private func expandedSharedHeads(_ states: MLXArray) -> MLXArray {
        tiled(states, repetitions: [1, layout.sharedHeadGroupSize, 1, 1])
    }
}

private final class PlamoFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: PlamoConfiguration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class PlamoDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: PlamoAttention
    @ModuleInfo(key: "mlp") private var feedForward: PlamoFeedForward
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: PlamoConfiguration) {
        _attention.wrappedValue = PlamoAttention(config)
        _feedForward.wrappedValue = PlamoFeedForward(config)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normalized = norm(hiddenStates)
        return hiddenStates
            + attention(normalized, mask: mask, cache: cache)
            + feedForward(normalized)
    }
}

private final class PlamoDecoder: Module {
    @ModuleInfo(key: "layers") fileprivate var layers: [PlamoDecoderLayer]

    init(_ config: PlamoConfiguration) {
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            PlamoDecoderLayer(config)
        }
    }
}

private final class PlamoBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var decoder: PlamoDecoder
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: PlamoConfiguration) {
        precondition(config.vocabularySize > 0, "Plamo vocab_size must be positive")
        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _decoder.wrappedValue = PlamoDecoder(config)
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbeddings(inputs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in decoder.layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }

        return finalNorm(hiddenStates)
    }
}

internal final class PlamoModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let configuration: PlamoConfiguration

    @ModuleInfo(key: "model") private var model: PlamoBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: PlamoConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.keyValueHeads, count: config.hiddenLayers)
        _model.wrappedValue = PlamoBackbone(config)
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
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

extension PlamoModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.decoder.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
