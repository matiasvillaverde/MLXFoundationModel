import Foundation
import MLX
import MLXNN

internal struct Olmo3LayerSchedule: Equatable, Sendable {
    internal static let fullAttention = "full_attention"
    internal static let slidingAttention = "sliding_attention"

    internal let types: [String]
    internal let firstSlidingIndex: Int
    internal let firstFullIndex: Int

    internal init(types: [String]) {
        precondition(!types.isEmpty, "layer_types must not be empty")

        self.types = types
        self.firstSlidingIndex = types.firstIndex(of: Self.slidingAttention) ?? 0
        self.firstFullIndex = types.firstIndex(of: Self.fullAttention) ?? firstSlidingIndex
    }

    internal static func defaultTypes(layerCount: Int) -> [String] {
        precondition(layerCount > 0, "num_hidden_layers must be positive")
        return (0 ..< layerCount).map { index in
            (index + 1).isMultiple(of: 4) ? Self.fullAttention : Self.slidingAttention
        }
    }

    internal func usesFullAttention(layerIndex: Int) -> Bool {
        types[layerIndex] == Self.fullAttention
    }
}

internal struct Olmo3AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float
    internal let usesFullAttention: Bool

    internal init(_ config: Olmo3Configuration, layerType: String) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")

        let headSize = Self.resolveHeadSize(config)
        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = headSize
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
        self.usesFullAttention = layerType == Olmo3LayerSchedule.fullAttention
    }

    private static func resolveHeadSize(_ config: Olmo3Configuration) -> Int {
        if let headDimensions = config.headDimensions {
            precondition(headDimensions > 0, "head_dim must be positive")
            return headDimensions
        }

        precondition(
            config.hiddenSize % config.attentionHeads == 0,
            "hidden_size must be divisible by num_attention_heads when head_dim is absent"
        )
        return config.hiddenSize / config.attentionHeads
    }
}

private final class Olmo3Attention: Module {
    private let layout: Olmo3AttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") private var keyNorm: RMSNorm

    init(_ config: Olmo3Configuration, layerType: String) {
        let layout = Olmo3AttentionLayout(config, layerType: layerType)
        self.layout = layout
        self.rope = Self.makeRoPE(config, layout: layout)

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
        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: layout.queryProjectionSize,
            eps: config.rmsNormEps
        )
        self._keyNorm.wrappedValue = RMSNorm(
            dimensions: layout.keyValueProjectionSize,
            eps: config.rmsNormEps
        )
    }

    private static func makeRoPE(
        _ config: Olmo3Configuration,
        layout: Olmo3AttentionLayout
    ) -> RoPELayer {
        guard layout.usesFullAttention else {
            return RoPE(
                dimensions: layout.headSize,
                traditional: false,
                base: config.ropeTheta
            )
        }

        return initializeRope(
            dims: layout.headSize,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)

        var queries = queryNorm(queryProjection(input))
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = keyNorm(keyProjection(input))
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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

private final class Olmo3FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: Olmo3Configuration) {
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

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class Olmo3Block: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: Olmo3Attention
    @ModuleInfo(key: "mlp") private var feedForward: Olmo3FeedForward

    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") private var postFeedForwardNorm: RMSNorm

    init(_ config: Olmo3Configuration, layerType: String) {
        self._attention.wrappedValue = Olmo3Attention(config, layerType: layerType)
        self._feedForward.wrappedValue = Olmo3FeedForward(config)
        self._postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postFeedForwardNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = input
            + postAttentionNorm(attention(input, mask: mask, cache: cache))
        return afterAttention + postFeedForwardNorm(feedForward(afterAttention))
    }
}

internal final class Olmo3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embeddings: Embedding
    @ModuleInfo fileprivate var layers: [Olmo3Block]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    private let schedule: Olmo3LayerSchedule
    private let slidingWindow: Int

    init(_ config: Olmo3Configuration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        let schedule = Olmo3LayerSchedule(types: config.layerTypes)
        self.schedule = schedule
        self.slidingWindow = config.slidingWindow
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = schedule.types.map { layerType in
            Olmo3Block(config, layerType: layerType)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embeddings(tokens)
        let fullMask = createAttentionMask(
            h: hiddenStates,
            cache: cache?[schedule.firstFullIndex]
        )
        let slidingMask = createAttentionMask(
            h: hiddenStates,
            cache: cache?[schedule.firstSlidingIndex],
            windowSize: slidingWindow
        )

        for (index, layer) in layers.enumerated() {
            let mask = schedule.usesFullAttention(layerIndex: index) ? fullMask : slidingMask
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }

        return norm(hiddenStates)
    }
}

internal final class Olmo3Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: Olmo3Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    private let configuration: Olmo3Configuration

    internal init(_ configuration: Olmo3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self._model.wrappedValue = Olmo3Backbone(configuration)

        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
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

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { layerType in
            layerType == Olmo3LayerSchedule.fullAttention
                ? KVCacheSimple()
                : RotatingKVCache(maxSize: configuration.slidingWindow)
        }
    }
}

internal struct Olmo3Configuration: Codable, Sendable, Equatable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var ropeTheta: Float
    var attentionBias: Bool
    var layerTypes: [String]
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    internal init(
        hiddenSize: Int = 4_096,
        hiddenLayers: Int = 32,
        intermediateSize: Int = 11_008,
        attentionHeads: Int = 32,
        headDimensions: Int? = nil,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 100_278,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int = 65_536,
        slidingWindow: Int = 4_096,
        ropeTheta: Float = 500_000,
        attentionBias: Bool = false,
        layerTypes: [String]? = nil,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false
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
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.attentionBias = attentionBias
        self.layerTypes = Self.resolvedLayerTypes(layerTypes, hiddenLayers: hiddenLayers)
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    internal var headSize: Int {
        if let headDimensions {
            return headDimensions
        }
        return hiddenSize / attentionHeads
    }

    private static func resolvedLayerTypes(
        _ layerTypes: [String]?,
        hiddenLayers: Int
    ) -> [String] {
        guard let layerTypes, !layerTypes.isEmpty else {
            return Olmo3LayerSchedule.defaultTypes(layerCount: hiddenLayers)
        }
        precondition(
            layerTypes.count == hiddenLayers,
            "layer_types count must match num_hidden_layers"
        )
        return layerTypes
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
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case layerTypes = "layer_types"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        let attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32

        self.init(
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4_096,
            hiddenLayers: hiddenLayers,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 11_008,
            attentionHeads: attentionHeads,
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 100_278,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 65_536,
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
                ?? 4_096,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 500_000,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            layerTypes: try container.decodeIfPresent([String].self, forKey: .layerTypes),
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

extension Olmo3Model: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
