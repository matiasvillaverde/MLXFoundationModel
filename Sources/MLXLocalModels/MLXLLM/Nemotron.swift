import Foundation
import MLX
import MLXFast
import MLXNN

internal struct NemotronConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenActivation: String
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var normEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var attentionBias: Bool
    var mlpBias: Bool
    var partialRotaryFactor: Float
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "nemotron",
        hiddenSize: Int,
        hiddenActivation: String = "relu2",
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        normEps: Float = 1e-5,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        headDimensions: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        partialRotaryFactor: Float = 0.5,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenActivation = hiddenActivation
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.normEps = normEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.headDimensions = headDimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? hiddenSize / attentionHeads
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenActivation = "hidden_act"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case normEps = "norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeScaling
        )
        try Self.validate(ropeScaling: ropeScaling, in: container)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "nemotron",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenActivation: try container.decodeIfPresent(
                String.self,
                forKey: .hiddenActivation
            ) ?? "relu2",
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            normEps: try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            attentionBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .attentionBias
            ) ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 0.5,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }

    private static func validate(
        ropeScaling: [String: StringOrNumber]?,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws {
        guard let ropeScaling else {
            return
        }
        guard ropeScaling["factor"]?.asFloat() != nil else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeScaling,
                in: container,
                debugDescription: "rope_scaling must contain numeric 'factor'"
            )
        }
        guard let value = ropeScaling["type"] ?? ropeScaling["rope_type"],
              case .string(let type) = value else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeScaling,
                in: container,
                debugDescription: "rope_scaling must contain string 'type' or 'rope_type'"
            )
        }
        guard type == "linear" else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeScaling,
                in: container,
                debugDescription: "Nemotron rope_scaling currently supports only 'linear'"
            )
        }
    }
}

internal struct NemotronAttentionLayout: Equatable, Sendable {
    let hiddenSize: Int
    let queryHeads: Int
    let keyValueHeads: Int
    let headDimensions: Int
    let rotaryDimensions: Int
    let queryProjectionSize: Int
    let keyValueProjectionSize: Int
    let attentionScale: Float
    let ropeScale: Float

    init(_ config: NemotronConfiguration) {
        precondition(config.hiddenSize > 0, "Nemotron hidden size must be positive")
        precondition(config.attentionHeads > 0, "Nemotron attention heads must be positive")
        precondition(config.kvHeads > 0, "Nemotron KV heads must be positive")
        if config.headDimensions == nil {
            precondition(
                config.hiddenSize.isMultiple(of: config.attentionHeads),
                "Nemotron hidden size must divide evenly across attention heads"
            )
        }
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "Nemotron attention heads must group KV heads"
        )
        precondition(
            config.partialRotaryFactor > 0 && config.partialRotaryFactor <= 1,
            "Nemotron partial rotary factor must be in (0, 1]"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = config.resolvedHeadDimensions
        self.rotaryDimensions = max(1, Int(Float(headDimensions) * config.partialRotaryFactor))
        self.queryProjectionSize = queryHeads * headDimensions
        self.keyValueProjectionSize = keyValueHeads * headDimensions
        self.attentionScale = pow(Float(headDimensions), -0.5)

        if let factor = config.ropeScaling?["factor"]?.asFloat() {
            self.ropeScale = 1 / factor
        } else {
            self.ropeScale = 1
        }
    }
}

internal final class NemotronLayerNorm1P: Module, UnaryLayer {
    let eps: Float

    @ParameterInfo(key: "weight") private var weight: MLXArray
    @ParameterInfo(key: "bias") private var bias: MLXArray

    init(dimensions: Int, eps: Float) {
        precondition(dimensions > 0, "Nemotron LayerNorm dimensions must be positive")
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self._bias.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        MLXFast.layerNorm(hiddenStates, weight: weight + 1, bias: bias, eps: eps)
    }
}

private final class NemotronSelfAttention: Module {
    private let layout: NemotronAttentionLayout

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    private let rope: RoPE

    init(_ config: NemotronConfiguration) {
        let layout = NemotronAttentionLayout(config)
        self.layout = layout
        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: config.attentionBias
        )
        self.rope = RoPE(
            dimensions: layout.rotaryDimensions,
            traditional: config.ropeTraditional,
            base: config.ropeTheta,
            scale: layout.ropeScale
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
        let values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
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

private final class NemotronFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: NemotronConfiguration) {
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(reluSquared(upProjection(hiddenStates)))
    }
}

private final class NemotronTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: NemotronSelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: NemotronFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: NemotronLayerNorm1P
    @ModuleInfo(key: "post_attention_layernorm")
    private var postAttentionLayerNorm: NemotronLayerNorm1P

    init(_ config: NemotronConfiguration) {
        self._selfAttention.wrappedValue = NemotronSelfAttention(config)
        self._feedForward.wrappedValue = NemotronFeedForward(config)
        self._inputLayerNorm.wrappedValue = NemotronLayerNorm1P(
            dimensions: config.hiddenSize,
            eps: config.normEps
        )
        self._postAttentionLayerNorm.wrappedValue = NemotronLayerNorm1P(
            dimensions: config.hiddenSize,
            eps: config.normEps
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

private final class NemotronBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [NemotronTransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: NemotronLayerNorm1P

    init(_ config: NemotronConfiguration) {
        precondition(config.vocabularySize > 0, "Nemotron vocab size must be positive")
        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            NemotronTransformerBlock(config)
        }
        self._finalNorm.wrappedValue = NemotronLayerNorm1P(
            dimensions: config.hiddenSize,
            eps: config.normEps
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

internal final class NemotronModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let config: NemotronConfiguration
    fileprivate let model: NemotronBackbone

    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: NemotronConfiguration) {
        self.config = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = NemotronBackbone(config)
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
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
        if config.tieWordEmbeddings {
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

extension NemotronModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
