import Darwin
import MLX
import MLXFast
import MLXNN

internal struct InternLM2RoPEPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let maxPositionEmbeddings: Int
    internal let traditional: Bool
    internal let base: Float
    internal let positionScale: Float
    internal let dynamicFactor: Float?

    internal init(_ config: InternLM2Configuration, dimensions: Int) {
        precondition(dimensions > 0, "rotary dimensions must be positive")
        precondition(config.maxPositionEmbeddings > 0, "max_position_embeddings must be positive")

        self.dimensions = dimensions
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.traditional = config.ropeTraditional
        self.base = config.ropeTheta

        let ropeType = Self.ropeType(config.ropeScaling)
        let factor = config.ropeScaling?["factor"]?.asFloat() ?? 1
        precondition(factor > 0, "rope_scaling.factor must be positive")

        switch ropeType {
        case "linear":
            self.positionScale = 1 / factor
            self.dynamicFactor = nil
        case "dynamic":
            precondition(dimensions > 2, "dynamic RoPE scaling requires dimensions greater than 2")
            self.positionScale = 1
            self.dynamicFactor = factor
        default:
            self.positionScale = 1
            self.dynamicFactor = nil
        }
    }

    internal func adjustedBase(sequenceLength: Int) -> Float {
        guard let dynamicFactor, sequenceLength > maxPositionEmbeddings else {
            return base
        }

        let scaledLength = dynamicFactor * Float(sequenceLength) / Float(maxPositionEmbeddings)
        let ratio = scaledLength - (dynamicFactor - 1)
        let exponent = Float(dimensions) / Float(dimensions - 2)
        return base * Darwin.powf(ratio, exponent)
    }

    private static func ropeType(_ scaling: [String: StringOrNumber]?) -> String? {
        guard let value = scaling?["type"] ?? scaling?["rope_type"],
              case .string(let string) = value else {
            return nil
        }
        return string
    }
}

private final class InternLM2RotaryEmbedding: Module {
    private let plan: InternLM2RoPEPlan

    init(_ plan: InternLM2RoPEPlan) {
        self.plan = plan
    }

    func callAsFunction(_ input: MLXArray, offset: Int = 0) -> MLXArray {
        let sequenceLength = input.dim(-2) + offset
        return MLXFast.RoPE(
            input,
            dimensions: plan.dimensions,
            traditional: plan.traditional,
            base: plan.adjustedBase(sequenceLength: sequenceLength),
            scale: plan.positionScale,
            offset: offset
        )
    }
}

internal struct InternLM2AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let keyValueGroups: Int
    internal let headSize: Int
    internal let packedProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: InternLM2Configuration) {
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

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.keyValueGroups = config.attentionHeads / config.kvHeads
        self.headSize = config.hiddenSize / config.attentionHeads
        self.packedProjectionSize = (queryHeads + 2 * keyValueHeads) * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

private final class InternLM2PackedAttention: Module {
    private let layout: InternLM2AttentionLayout

    @ModuleInfo(key: "wqkv") private var queryKeyValueProjection: Linear
    @ModuleInfo(key: "wo") private var outputProjection: Linear

    private let rope: InternLM2RotaryEmbedding

    init(_ config: InternLM2Configuration) {
        let layout = InternLM2AttentionLayout(config)
        self.layout = layout
        self._queryKeyValueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.packedProjectionSize,
            bias: config.bias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryHeads * layout.headSize,
            layout.hiddenSize,
            bias: config.bias
        )
        self.rope = InternLM2RotaryEmbedding(
            InternLM2RoPEPlan(config, dimensions: layout.headSize)
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        var packed = queryKeyValueProjection(hiddenStates)
        packed = packed.reshaped(
            batchSize,
            tokenCount,
            layout.keyValueHeads,
            2 + layout.keyValueGroups,
            layout.headSize
        )

        var queries = packed[.ellipsis, ..<layout.keyValueGroups, 0...]
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = packed[.ellipsis, -2, 0...]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = packed[.ellipsis, -1, 0...]
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

private final class InternLM2FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "w1") private var gateProjection: Linear
    @ModuleInfo(key: "w2") private var downProjection: Linear
    @ModuleInfo(key: "w3") private var upProjection: Linear

    init(_ config: InternLM2Configuration) {
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
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class InternLM2TransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: InternLM2PackedAttention
    @ModuleInfo(key: "feed_forward") private var feedForward: InternLM2FeedForward
    @ModuleInfo(key: "attention_norm") private var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") private var feedForwardNorm: RMSNorm

    init(_ config: InternLM2Configuration) {
        self._attention.wrappedValue = InternLM2PackedAttention(config)
        self._feedForward.wrappedValue = InternLM2FeedForward(config)
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._feedForwardNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = attention(attentionNorm(hiddenStates), mask: mask, cache: cache)
        let afterAttention = hiddenStates + attentionOutput
        return afterAttention + feedForward(feedForwardNorm(afterAttention))
    }
}

private final class InternLM2Backbone: Module {
    @ModuleInfo(key: "tok_embeddings") var tokenEmbeddings: Embedding
    @ModuleInfo var layers: [InternLM2TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    init(_ config: InternLM2Configuration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            InternLM2TransformerBlock(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        let mask = createAttentionMask(h: tokens, cache: cache)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hiddenStates)
    }
}

internal final class InternLM2Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: InternLM2Backbone

    @ModuleInfo(key: "output") private var output: Linear?

    public init(_ config: InternLM2Configuration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = InternLM2Backbone(config)
        if !config.tieWordEmbeddings {
            self._output.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hiddenStates = model(inputs, cache: cache)
        if let output {
            return output(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        if let output {
            return greedyTokenOutput(logits: output(hiddenStates), state: state)
        }
        return greedyTokenOutput(logits: model.tokenEmbeddings.asLinear(hiddenStates), state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("attention.rope.inv_freq") }
    }
}

extension InternLM2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["wqkv"]) }
    }
}

internal struct InternLM2Configuration: Codable, Sendable, Equatable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var bias: Bool

    internal init(
        hiddenSize: Int = 4_096,
        hiddenLayers: Int = 32,
        intermediateSize: Int = 14_336,
        attentionHeads: Int = 32,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int = 92_544,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int = 32_768,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        bias: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.bias = bias
    }

    var kvGroups: Int {
        attentionHeads / kvHeads
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case bias = "bias"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeScaling
        )

        try Self.validate(ropeScaling: ropeScaling, in: container)

        self.init(
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 32_768,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            bias: try container.decodeIfPresent(Bool.self, forKey: .bias) ?? true
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
                debugDescription: "rope_scaling.factor must be numeric"
            )
        }

        guard let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"],
              case .string(let string) = ropeType,
              string == "linear" || string == "dynamic" else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeScaling,
                in: container,
                debugDescription: "rope_scaling type must be 'linear' or 'dynamic'"
            )
        }
    }
}
