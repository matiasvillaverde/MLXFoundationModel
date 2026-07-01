import Foundation
import MLX
import MLXNN

internal struct Cohere2LayerSchedule: Equatable, Sendable {
    internal let layerCount: Int
    internal let slidingWindowPattern: Int
    internal let slidingWindow: Int
    internal let firstSlidingLayerIndex: Int?
    internal let firstFullLayerIndex: Int?

    internal init(_ config: Cohere2Configuration) {
        precondition(config.hiddenLayers > 0, "num_hidden_layers must be positive")
        precondition(config.slidingWindowPattern > 0, "sliding_window_pattern must be positive")
        precondition(config.slidingWindow > 0, "sliding_window must be positive")

        self.layerCount = config.hiddenLayers
        self.slidingWindowPattern = config.slidingWindowPattern
        self.slidingWindow = config.slidingWindow
        self.firstSlidingLayerIndex = Self.firstIndex(
            layerCount: config.hiddenLayers,
            matching: { Self.usesSlidingWindow($0, pattern: config.slidingWindowPattern) }
        )
        self.firstFullLayerIndex = Self.firstIndex(
            layerCount: config.hiddenLayers,
            matching: { !Self.usesSlidingWindow($0, pattern: config.slidingWindowPattern) }
        )
    }

    internal func usesSlidingWindow(layerIndex: Int) -> Bool {
        Self.usesSlidingWindow(layerIndex, pattern: slidingWindowPattern)
    }

    internal static func defaultLayerTypes(layerCount: Int, pattern: Int) -> [String] {
        (0 ..< layerCount).map { layerIndex in
            usesSlidingWindow(layerIndex, pattern: pattern)
                ? "sliding_attention"
                : "full_attention"
        }
    }

    private static func usesSlidingWindow(_ layerIndex: Int, pattern: Int) -> Bool {
        (layerIndex + 1).isMultiple(of: pattern) == false
    }

    private static func firstIndex(
        layerCount: Int,
        matching predicate: (Int) -> Bool
    ) -> Int? {
        (0 ..< layerCount).first(where: predicate)
    }
}

internal struct Cohere2AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float
    internal let usesSlidingWindow: Bool

    internal init(_ config: Cohere2Configuration, usesSlidingWindow: Bool) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")

        let headSize = config.headDimensions ?? config.hiddenSize / config.attentionHeads
        precondition(headSize > 0, "head_dim must be positive")
        precondition(
            headSize * config.attentionHeads == config.hiddenSize,
            "head_dim * num_attention_heads must equal hidden_size"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = headSize
        self.queryProjectionSize = config.attentionHeads * headSize
        self.keyValueProjectionSize = config.kvHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
        self.usesSlidingWindow = usesSlidingWindow
    }
}

private final class Cohere2Attention: Module {
    private let layout: Cohere2AttentionLayout
    private let rope: RoPE

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: Cohere2Configuration, usesSlidingWindow: Bool) {
        let layout = Cohere2AttentionLayout(config, usesSlidingWindow: usesSlidingWindow)
        self.layout = layout
        self.rope = RoPE(
            dimensions: layout.headSize,
            traditional: true,
            base: config.ropeTheta
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

        if layout.usesSlidingWindow {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        return outputProjection(
            attentionWithCacheUpdate(
                queries: queries.asType(.float32),
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.attentionScale,
                mask: mask
            )
            .asType(values.dtype)
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, layout.queryProjectionSize)
        )
    }
}

private final class Cohere2FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: Cohere2Configuration) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class Cohere2Block: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: Cohere2Attention
    @ModuleInfo(key: "mlp") private var feedForward: Cohere2FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: LayerNorm

    init(_ config: Cohere2Configuration, usesSlidingWindow: Bool) {
        self._attention.wrappedValue = Cohere2Attention(
            config,
            usesSlidingWindow: usesSlidingWindow
        )
        self._feedForward.wrappedValue = Cohere2FeedForward(config)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps,
            bias: config.layerNormBias
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normalizedStates = inputLayerNorm(hiddenStates)
        return hiddenStates
            + attention(normalizedStates, mask: mask, cache: cache)
            + feedForward(normalizedStates)
    }
}

private final class Cohere2Backbone: Module {
    private let schedule: Cohere2LayerSchedule

    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [Cohere2Block]
    @ModuleInfo(key: "norm") private var norm: LayerNorm

    init(_ config: Cohere2Configuration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        let schedule = Cohere2LayerSchedule(config)
        self.schedule = schedule
        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            Cohere2Block(
                config,
                usesSlidingWindow: schedule.usesSlidingWindow(layerIndex: layerIndex)
            )
        }
        self._norm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps,
            bias: config.layerNormBias
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)

        let fullMask = schedule.firstFullLayerIndex.map {
            createAttentionMask(h: hiddenStates, cache: cache?[$0])
        } ?? MLXFast.ScaledDotProductAttentionMaskMode.none
        let slidingMask = schedule.firstSlidingLayerIndex.map {
            createAttentionMask(
                h: hiddenStates,
                cache: cache?[$0],
                windowSize: schedule.slidingWindow
            )
        } ?? MLXFast.ScaledDotProductAttentionMaskMode.none

        for (layerIndex, layer) in layers.enumerated() {
            let mask = schedule.usesSlidingWindow(layerIndex: layerIndex)
                ? slidingMask
                : fullMask
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hiddenStates)
    }
}

internal final class Cohere2Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: Cohere2Backbone

    private let config: Cohere2Configuration

    internal init(_ config: Cohere2Configuration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self._model.wrappedValue = Cohere2Backbone(config)
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

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let schedule = Cohere2LayerSchedule(config)
        return (0 ..< config.hiddenLayers).map { layerIndex in
            schedule.usesSlidingWindow(layerIndex: layerIndex)
                ? RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
                : KVCacheSimple()
        }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        model.tokenEmbeddings.asLinear(hiddenStates) * config.logitScale
    }
}

internal struct Cohere2Configuration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var kvHeads: Int
    internal var headDimensions: Int?
    internal var ropeTheta: Float
    internal var vocabularySize: Int
    internal var layerNormEps: Float
    internal var logitScale: Float
    internal var attentionBias: Bool
    internal var layerNormBias: Bool
    internal var slidingWindow: Int
    internal var slidingWindowPattern: Int
    internal var layerTypes: [String]

    internal init(
        modelType: String = "cohere2",
        hiddenSize: Int = 4_096,
        hiddenLayers: Int = 32,
        intermediateSize: Int = 14_336,
        attentionHeads: Int = 32,
        kvHeads: Int = 8,
        headDimensions: Int? = 128,
        ropeTheta: Float = 50_000,
        vocabularySize: Int = 256_000,
        layerNormEps: Float = 1e-5,
        logitScale: Float = 0.0625,
        attentionBias: Bool = false,
        layerNormBias: Bool = false,
        slidingWindow: Int = 4_096,
        slidingWindowPattern: Int = 4,
        layerTypes: [String]? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDimensions = headDimensions
        self.ropeTheta = ropeTheta
        self.vocabularySize = vocabularySize
        self.layerNormEps = layerNormEps
        self.logitScale = logitScale
        self.attentionBias = attentionBias
        self.layerNormBias = layerNormBias
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.layerTypes = layerTypes ?? Cohere2LayerSchedule.defaultLayerTypes(
            layerCount: hiddenLayers,
            pattern: slidingWindowPattern
        )
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case layerNormEps = "layer_norm_eps"
        case logitScale = "logit_scale"
        case attentionBias = "attention_bias"
        case layerNormBias = "layer_norm_bias"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case underscoredSlidingWindowPattern = "_sliding_window_pattern"
        case layerSwitch = "layer_switch"
        case layerTypes = "layer_types"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try container.decodeIfPresent(
            Int.self,
            forKey: .hiddenLayers
        ) ?? 32
        let slidingWindowPattern = try Self.decodeInt(
            from: container,
            keys: [.slidingWindowPattern, .underscoredSlidingWindowPattern, .layerSwitch],
            default: 4
        )

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "cohere2",
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? 4_096,
            hiddenLayers: hiddenLayers,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 14_336,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? 32,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8,
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 50_000,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 256_000,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
                ?? 1e-5,
            logitScale: try container.decodeIfPresent(Float.self, forKey: .logitScale)
                ?? 0.0625,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            layerNormBias: try container.decodeIfPresent(Bool.self, forKey: .layerNormBias)
                ?? false,
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
                ?? 4_096,
            slidingWindowPattern: slidingWindowPattern,
            layerTypes: try container.decodeIfPresent([String].self, forKey: .layerTypes)
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encodeIfPresent(headDimensions, forKey: .headDimensions)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(layerNormEps, forKey: .layerNormEps)
        try container.encode(logitScale, forKey: .logitScale)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(layerNormBias, forKey: .layerNormBias)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(slidingWindowPattern, forKey: .slidingWindowPattern)
        try container.encode(layerTypes, forKey: .layerTypes)
    }

    private static func decodeInt(
        from container: KeyedDecodingContainer<CodingKeys>,
        keys: [CodingKeys],
        default defaultValue: Int
    ) throws -> Int {
        for key in keys {
            if let value = try container.decodeIfPresent(Int.self, forKey: key) {
                return value
            }
        }
        return defaultValue
    }
}

extension Cohere2Model: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
