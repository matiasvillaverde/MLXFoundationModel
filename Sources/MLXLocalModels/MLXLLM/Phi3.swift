import MLX
import MLXFast
import MLXNN

internal struct Phi3AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let rotaryDimensions: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let packedProjectionSize: Int
    internal let keySplitIndex: Int
    internal let valueSplitIndex: Int
    internal let attentionScale: Float

    internal init(_ config: Phi3Configuration) {
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
        precondition(config.partialRotaryFactor > 0, "partial_rotary_factor must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = config.hiddenSize / config.attentionHeads
        self.rotaryDimensions = min(
            headSize,
            max(1, Int(Float(headSize) * config.partialRotaryFactor))
        )
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.packedProjectionSize = queryProjectionSize + 2 * keyValueProjectionSize
        self.keySplitIndex = queryProjectionSize
        self.valueSplitIndex = queryProjectionSize + keyValueProjectionSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

internal struct Phi3RotaryPlan: Equatable, Sendable {
    internal enum Kind: Equatable, Sendable {
        case rope(scale: Float)
        case longRoPE(longFactor: [Float])
    }

    internal let dimensions: Int
    internal let base: Float
    internal let traditional: Bool
    internal let maxPositionEmbeddings: Int
    internal let originalMaxPositionEmbeddings: Int
    internal let kind: Kind

    internal init(_ config: Phi3Configuration, layout: Phi3AttentionLayout) {
        precondition(config.maxPositionEmbeddings > 0, "max_position_embeddings must be positive")
        precondition(
            config.originalMaxPositionEmbeddings > 0,
            "original_max_position_embeddings must be positive"
        )

        self.dimensions = layout.rotaryDimensions
        self.base = config.ropeTheta
        self.traditional = config.ropeTraditional
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = config.originalMaxPositionEmbeddings

        if let scaling = config.ropeScaling,
           scaling.usesLongRoPE,
           let longFactor = scaling.longFactor,
           !longFactor.isEmpty {
            self.kind = .longRoPE(longFactor: longFactor)
        } else {
            let scale: Float
            if config.ropeScaling?.type == "linear",
               let factor = config.ropeScaling?.factor,
               factor > 0 {
                scale = 1 / factor
            } else {
                scale = 1
            }
            self.kind = .rope(scale: scale)
        }
    }
}

private enum Phi3RotaryEmbedding {
    case rope(RoPE)
    case longRoPE(SuScaledRotaryEmbedding)

    init(_ plan: Phi3RotaryPlan) {
        switch plan.kind {
        case .rope(let scale):
            self = .rope(
                RoPE(
                    dimensions: plan.dimensions,
                    traditional: plan.traditional,
                    base: plan.base,
                    scale: scale
                )
            )
        case .longRoPE(let longFactor):
            self = .longRoPE(
                SuScaledRotaryEmbedding(
                    dimensions: plan.dimensions,
                    base: plan.base,
                    maxPositionEmbeddings: plan.maxPositionEmbeddings,
                    originalMaxPositionEmbeddings: plan.originalMaxPositionEmbeddings,
                    longFactor: longFactor
                )
            )
        }
    }

    func apply(to hiddenStates: MLXArray, offset: Int = 0) -> MLXArray {
        switch self {
        case .rope(let rope):
            rope(hiddenStates, offset: offset)
        case .longRoPE(let rope):
            rope(hiddenStates, offset: offset)
        }
    }
}

private final class Phi3PackedAttention: Module {
    private let layout: Phi3AttentionLayout

    @ModuleInfo(key: "qkv_proj") private var queryKeyValueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    private let rotaryEmbedding: Phi3RotaryEmbedding

    init(_ config: Phi3Configuration) {
        let layout = Phi3AttentionLayout(config)
        self.layout = layout
        self._queryKeyValueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.packedProjectionSize,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: false
        )
        self.rotaryEmbedding = Phi3RotaryEmbedding(Phi3RotaryPlan(config, layout: layout))
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        let packed = split(
            queryKeyValueProjection(hiddenStates),
            indices: [layout.keySplitIndex, layout.valueSplitIndex],
            axis: -1
        )
        var queries = packed[0]
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = packed[1]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = packed[2]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rotaryEmbedding.apply(to: queries, offset: cache.offset)
            keys = rotaryEmbedding.apply(to: keys, offset: cache.offset)
        } else {
            queries = rotaryEmbedding.apply(to: queries)
            keys = rotaryEmbedding.apply(to: keys)
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

private final class Phi3FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") private var gateUpProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: Phi3Configuration) {
        self._gateUpProjection.wrappedValue = Linear(
            config.hiddenSize,
            2 * config.intermediateSize,
            bias: false
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let splitStates = split(gateUpProjection(hiddenStates), parts: 2, axis: -1)
        return downProjection(silu(splitStates[0]) * splitStates[1])
    }
}

private final class Phi3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Phi3PackedAttention
    @ModuleInfo(key: "mlp") private var feedForward: Phi3FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: Phi3Configuration) {
        self._selfAttention.wrappedValue = Phi3PackedAttention(config)
        self._feedForward.wrappedValue = Phi3FeedForward(config)
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
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = hiddenStates
            + selfAttention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class Phi3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var tokenEmbeddings: Embedding
    @ModuleInfo var layers: [Phi3TransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: Phi3Configuration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            Phi3TransformerBlock(config)
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        let mask = createAttentionMask(h: tokens, cache: cache)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

internal final class Phi3Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: Phi3Backbone
    private let tieWordEmbeddings: Bool

    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public init(_ config: Phi3Configuration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = Phi3Backbone(config)
        self.tieWordEmbeddings = config.tieWordEmbeddings

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
        }
    }

    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(tokens, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if tieWordEmbeddings {
            return model.tokenEmbeddings.asLinear(hiddenStates)
        }
        guard let lmHead else {
            fatalError("Phi3 requires either tied embeddings or lm_head weights")
        }
        return lmHead(hiddenStates)
    }
}

internal struct Phi3RoPEScaling: Codable, Sendable, Equatable {
    let longFactor: [Float]?
    let shortFactor: [Float]?
    let factor: Float?
    let type: String?
    let longMScale: Float?
    let shortMScale: Float?

    var usesLongRoPE: Bool {
        type == "su" || type == "longrope"
    }

    enum CodingKeys: String, CodingKey {
        case type
        case factor
        case longFactor = "long_factor"
        case shortFactor = "short_factor"
        case longMScale = "long_mscale"
        case shortMScale = "short_mscale"
    }
}

internal typealias RopeScalingWithFactorArrays = Phi3RoPEScaling

internal struct Phi3Configuration: Codable, Sendable, Equatable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: Phi3RoPEScaling?
    var partialRotaryFactor: Float
    var maxPositionEmbeddings: Int
    var originalMaxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool

    internal init(
        hiddenSize: Int = 3_072,
        hiddenLayers: Int = 32,
        intermediateSize: Int = 8_192,
        attentionHeads: Int = 32,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int = 32_064,
        kvHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: Phi3RoPEScaling? = nil,
        partialRotaryFactor: Float = 1,
        maxPositionEmbeddings: Int = 4_096,
        originalMaxPositionEmbeddings: Int? = nil,
        tieWordEmbeddings: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.partialRotaryFactor = partialRotaryFactor
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
            ?? maxPositionEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let attentionHeads = try container.decodeIfPresent(
            Int.self,
            forKey: .attentionHeads
        ) ?? 32
        let maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self,
            forKey: .maxPositionEmbeddings
        ) ?? 4_096

        self.init(
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3_072,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 8_192,
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 32_064,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            ropeScaling: try container.decodeIfPresent(Phi3RoPEScaling.self, forKey: .ropeScaling),
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 1,
            maxPositionEmbeddings: maxPositionEmbeddings,
            originalMaxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .originalMaxPositionEmbeddings
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

extension Phi3Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["qkv_proj"]) }
    }
}
