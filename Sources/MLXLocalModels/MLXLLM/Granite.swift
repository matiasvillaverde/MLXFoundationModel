import Foundation
import MLX
import MLXNN

internal struct GraniteAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: GraniteConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(
            config.hiddenSize % config.attentionHeads == 0,
            "hidden_size must be divisible by num_attention_heads"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = config.hiddenSize / config.attentionHeads
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = config.attentionMultiplier
    }
}

internal struct GraniteRoPEPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let base: Float
    internal let scale: Float

    internal init(_ config: GraniteConfiguration, dimensions: Int) {
        self.dimensions = dimensions
        self.base = config.ropeTheta
        self.scale = Self.scale(from: config.ropeScaling)
    }

    private static func scale(from ropeScaling: [String: StringOrNumber]?) -> Float {
        guard
            let ropeScaling,
            (ropeScaling["type"] ?? ropeScaling["rope_type"]) == .string("linear")
        else {
            return 1
        }

        guard let factor = ropeScaling["factor"]?.asFloat(), factor > 0 else {
            preconditionFailure("linear Granite rope_scaling requires a positive factor")
        }

        return 1 / factor
    }
}

private final class GraniteRotaryEmbedding {
    private let plan: GraniteRoPEPlan
    private let rope: RoPE

    init(_ plan: GraniteRoPEPlan) {
        self.plan = plan
        self.rope = RoPE(
            dimensions: plan.dimensions,
            traditional: false,
            base: plan.base,
            scale: plan.scale
        )
    }

    func apply(to input: MLXArray, offset: Int = 0) -> MLXArray {
        rope(input, offset: offset)
    }
}

private final class GraniteAttention: Module {
    private let layout: GraniteAttentionLayout
    private let rope: GraniteRotaryEmbedding

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: GraniteConfiguration) {
        let layout = GraniteAttentionLayout(config)
        self.layout = layout
        self.rope = GraniteRotaryEmbedding(
            GraniteRoPEPlan(config, dimensions: layout.headSize)
        )

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
        queries = rope.apply(to: queries, offset: offset)
        keys = rope.apply(to: keys, offset: offset)

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

private final class GraniteFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: GraniteConfiguration) {
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

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class GraniteBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: GraniteAttention
    @ModuleInfo(key: "mlp") private var feedForward: GraniteFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    private let residualMultiplier: Float

    init(_ config: GraniteConfiguration) {
        self._attention.wrappedValue = GraniteAttention(config)
        self._feedForward.wrappedValue = GraniteFeedForward(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self.residualMultiplier = config.residualMultiplier
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
        let attentionResidual = input + attentionOutput * residualMultiplier
        return attentionResidual
            + feedForward(postAttentionLayerNorm(attentionResidual)) * residualMultiplier
    }
}

private final class GraniteBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embeddings: Embedding
    @ModuleInfo fileprivate var layers: [GraniteBlock]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    private let embeddingMultiplier: Float

    init(_ config: GraniteConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in GraniteBlock(config) }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self.embeddingMultiplier = config.embeddingMultiplier
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embeddings(tokens) * embeddingMultiplier
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

internal final class GraniteModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: GraniteBackbone
    private let configuration: GraniteConfiguration
    private let logitsScaling: Float

    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public init(_ configuration: GraniteConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(
            repeating: configuration.kvHeads,
            count: configuration.hiddenLayers
        )
        self._model.wrappedValue = GraniteBackbone(configuration)
        self.logitsScaling = configuration.logitsScaling

        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
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
        let projected = lmHead.map { $0(hiddenStates) }
            ?? model.embeddings.asLinear(hiddenStates)
        return projected / logitsScaling
    }
}

internal struct GraniteConfiguration: Codable, Sendable, Equatable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var logitsScaling: Float
    var attentionMultiplier: Float
    var embeddingMultiplier: Float
    var residualMultiplier: Float
    var maxPositionEmbeddings: Int
    var kvHeads: Int
    var attentionBias: Bool
    var mlpBias: Bool
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    internal init(
        hiddenSize: Int = 2_048,
        hiddenLayers: Int = 40,
        intermediateSize: Int = 8_192,
        attentionHeads: Int = 32,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int = 49_159,
        logitsScaling: Float = 8,
        attentionMultiplier: Float = 1.0 / 64.0,
        embeddingMultiplier: Float = 12,
        residualMultiplier: Float = 0.22,
        maxPositionEmbeddings: Int = 131_072,
        kvHeads: Int? = nil,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        ropeTheta: Float = 10_000_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.logitsScaling = logitsScaling
        self.attentionMultiplier = attentionMultiplier
        self.embeddingMultiplier = embeddingMultiplier
        self.residualMultiplier = residualMultiplier
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case logitsScaling = "logits_scaling"
        case attentionMultiplier = "attention_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case residualMultiplier = "residual_multiplier"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let attentionHeads = try container.decodeIfPresent(
            Int.self,
            forKey: .attentionHeads
        ) ?? 32

        self.init(
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 8_192,
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 49_159,
            logitsScaling: try container.decodeIfPresent(
                Float.self,
                forKey: .logitsScaling
            ) ?? 8,
            attentionMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .attentionMultiplier
            ) ?? 1.0 / 64.0,
            embeddingMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .embeddingMultiplier
            ) ?? 12,
            residualMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .residualMultiplier
            ) ?? 0.22,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            attentionBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .attentionBias
            ) ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true
        )
    }
}

extension GraniteModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
