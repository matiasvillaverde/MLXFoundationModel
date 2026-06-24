import Darwin
import MLX
import MLXFast
import MLXNN

internal struct LlamaRoPEPlan: Equatable, Sendable {
    internal enum Kind: Equatable, Sendable {
        case standard
        case linear(factor: Float)
        case dynamic(factor: Float)
        case llama3(
            factor: Float,
            lowFrequencyFactor: Float,
            highFrequencyFactor: Float,
            originalMaxPositionEmbeddings: Float
        )
    }

    internal let dimensions: Int
    internal let maxPositionEmbeddings: Int
    internal let traditional: Bool
    internal let base: Float
    internal let kind: Kind

    internal init(_ config: LlamaConfiguration, dimensions: Int) {
        precondition(dimensions > 0, "rotary dimensions must be positive")

        self.dimensions = dimensions
        self.maxPositionEmbeddings = config.maxPositionEmbeddings ?? 2_048
        self.traditional = config.ropeTraditional
        self.base = config.ropeTheta

        let ropeType = Self.ropeType(config.ropeScaling)
        let factor = config.ropeScaling?["factor"]?.asFloat() ?? 1
        precondition(factor > 0, "rope_scaling.factor must be positive")

        switch ropeType {
        case "linear":
            self.kind = .linear(factor: factor)
        case "dynamic":
            precondition(dimensions > 2, "dynamic RoPE scaling requires dimensions greater than 2")
            self.kind = .dynamic(factor: factor)
        case "llama3":
            self.kind = .llama3(
                factor: factor,
                lowFrequencyFactor: config.ropeScaling?["low_freq_factor"]?.asFloat() ?? 1,
                highFrequencyFactor: config.ropeScaling?["high_freq_factor"]?.asFloat() ?? 4,
                originalMaxPositionEmbeddings: config.ropeScaling?[
                    "original_max_position_embeddings"
                ]?.asFloat() ?? 8_192
            )
        default:
            self.kind = .standard
        }
    }

    internal var positionScale: Float {
        if case .linear(let factor) = kind {
            return 1 / factor
        }
        return 1
    }

    internal func adjustedBase(sequenceLength: Int) -> Float {
        guard case .dynamic(let factor) = kind,
              sequenceLength > maxPositionEmbeddings else {
            return base
        }

        let scaledLength = factor * Float(sequenceLength) / Float(maxPositionEmbeddings)
        let ratio = scaledLength - (factor - 1)
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

private final class LlamaRotaryEmbedding {
    private let plan: LlamaRoPEPlan
    private let frequencies: MLXArray?

    init(_ plan: LlamaRoPEPlan) {
        self.plan = plan
        self.frequencies = Self.frequencies(for: plan)
    }

    func callAsFunction(_ input: MLXArray, offset: Int = 0) -> MLXArray {
        let sequenceLength = input.dim(-2) + offset
        return MLXFast.RoPE(
            input,
            dimensions: plan.dimensions,
            traditional: plan.traditional,
            base: frequencies == nil ? plan.adjustedBase(sequenceLength: sequenceLength) : nil,
            scale: plan.positionScale,
            offset: offset,
            freqs: frequencies
        )
    }

    private static func frequencies(for plan: LlamaRoPEPlan) -> MLXArray? {
        guard case .llama3(
            let factor,
            let lowFrequencyFactor,
            let highFrequencyFactor,
            let originalMaxPositionEmbeddings
        ) = plan.kind else {
            return nil
        }

        let lowFrequencyWavelength = originalMaxPositionEmbeddings / lowFrequencyFactor
        let highFrequencyWavelength = originalMaxPositionEmbeddings / highFrequencyFactor

        let indices = MLXArray(stride(from: 0, to: plan.dimensions, by: 2))
        var frequencies = MLX.pow(plan.base, indices / Float(plan.dimensions))
        let wavelengths = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelengths .> MLXArray(lowFrequencyWavelength),
            frequencies * factor,
            frequencies
        )

        let isMediumFrequency = MLX.logicalAnd(
            wavelengths .> MLXArray(highFrequencyWavelength),
            wavelengths .< MLXArray(lowFrequencyWavelength)
        )
        let smoothFactors = (
            originalMaxPositionEmbeddings / wavelengths - lowFrequencyFactor
        ) / (highFrequencyFactor - lowFrequencyFactor)
        let smoothFrequencies = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        return MLX.where(isMediumFrequency, smoothFrequencies, frequencies)
    }
}

internal struct LlamaAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: LlamaConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")
        if config.headDimensions == nil {
            precondition(
                config.hiddenSize.isMultiple(of: config.attentionHeads),
                "hidden_size must be divisible by num_attention_heads"
            )
        }
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        precondition(config.resolvedHeadDimensions > 0, "head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = config.resolvedHeadDimensions
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

private final class LlamaSelfAttention: Module {
    private let layout: LlamaAttentionLayout

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    private let rope: LlamaRotaryEmbedding

    init(_ config: LlamaConfiguration) {
        let layout = LlamaAttentionLayout(config)
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
        self.rope = LlamaRotaryEmbedding(
            LlamaRoPEPlan(config, dimensions: layout.headSize)
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

private final class LlamaFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: LlamaConfiguration) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class LlamaTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: LlamaSelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: LlamaFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: LlamaConfiguration) {
        self._selfAttention.wrappedValue = LlamaSelfAttention(config)
        self._feedForward.wrappedValue = LlamaFeedForward(config)
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

private final class LlamaBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [LlamaTransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: LlamaConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            LlamaTransformerBlock(config)
        }
        self._finalNorm.wrappedValue = RMSNorm(
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

/// Shared implementation for Llama-family and Mistral text models.
internal final class LlamaModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: LlamaBackbone

    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public init(_ config: LlamaConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = LlamaBackbone(config)
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
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

internal struct LlamaConfiguration: Codable, Sendable, Equatable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var mlpBias: Bool

    internal init(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDimensions: Int? = nil,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false,
        mlpBias: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? hiddenSize / attentionHeads
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
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
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
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            attentionBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .attentionBias
            ) ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
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
        guard let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"],
              case .string(let type) = ropeType else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeScaling,
                in: container,
                debugDescription: "rope_scaling must contain string 'type' or 'rope_type'"
            )
        }
        let supportedTypes: Set<String> = ["linear", "dynamic", "llama3"]
        guard supportedTypes.contains(type) else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeScaling,
                in: container,
                debugDescription:
                    "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'"
            )
        }
    }
}

extension LlamaModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
