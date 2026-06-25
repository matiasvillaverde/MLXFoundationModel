import Foundation
import MLX
import MLXFast
import MLXNN

internal struct Mistral3AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: Mistral3TextConfiguration) {
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

internal struct Mistral3LayerSchedule: Equatable, Sendable {
    internal static let fullAttention = "full_attention"
    internal static let slidingAttention = "sliding_attention"

    internal let layerTypes: [String]
    internal let fullAttentionMaskLayerIndex: Int
    internal let slidingAttentionMaskLayerIndex: Int?

    internal init(_ config: Mistral3TextConfiguration) {
        precondition(config.hiddenLayers > 0, "num_hidden_layers must be positive")
        let types = config.layerTypes.isEmpty
            ? Array(repeating: Self.fullAttention, count: config.hiddenLayers)
            : config.layerTypes
        precondition(
            types.count == config.hiddenLayers,
            "layer_types count must match num_hidden_layers"
        )

        for type in types {
            precondition(
                type == Self.fullAttention || type == Self.slidingAttention,
                "unsupported Mistral3 layer type: \(type)"
            )
        }

        self.layerTypes = types
        self.fullAttentionMaskLayerIndex = types.firstIndex(of: Self.fullAttention) ?? 0
        self.slidingAttentionMaskLayerIndex = types.firstIndex(of: Self.slidingAttention)
    }

    internal func usesSlidingWindow(at layerIndex: Int) -> Bool {
        layerTypes[layerIndex] == Self.slidingAttention
    }
}

internal struct Mistral3AttentionScalePlan: Equatable, Sendable {
    internal let beta: Float?
    internal let originalMaxPositionEmbeddings: Int?

    internal init(_ config: Mistral3TextConfiguration) {
        self.beta = config.ropeParameters?["llama_4_scaling_beta"]?.asFloat()
        self.originalMaxPositionEmbeddings = config.ropeParameters?[
            "original_max_position_embeddings"
        ]?.asInt()
        if let originalMaxPositionEmbeddings {
            precondition(
                originalMaxPositionEmbeddings > 0,
                "original_max_position_embeddings must be positive"
            )
        }
    }

    internal var usesPositionScaling: Bool {
        beta != nil && originalMaxPositionEmbeddings != nil
    }

    internal func values(start: Int, count: Int, dtype: DType) -> MLXArray {
        precondition(start >= 0, "attention scale start must be non-negative")
        precondition(count >= 0, "attention scale count must be non-negative")
        guard let beta, let originalMaxPositionEmbeddings else {
            return MLXArray.ones([count, 1]).asType(dtype)
        }

        let positions = MLXArray(Int32(start) ..< Int32(start + count))
        let blockIndex = MLX.floor(positions.asType(.float32) / Float(originalMaxPositionEmbeddings))
        let scale = 1 + beta * MLX.log(1 + blockIndex)
        return scale[0..., .newAxis].asType(dtype)
    }
}

private final class Mistral3SelfAttention: Module {
    private let layout: Mistral3AttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: Mistral3TextConfiguration) {
        let layout = Mistral3AttentionLayout(config)
        self.layout = layout
        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: false
        )

        self.rope = initializeRope(
            dims: layout.headSize,
            base: config.ropeParameters?["rope_theta"]?.asFloat() ?? config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeParameters,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionScale: MLXArray,
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

        queries = applyRotaryPosition(rope, to: queries, cache: cache) * attentionScale
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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

private final class Mistral3FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: Mistral3TextConfiguration) {
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

private final class Mistral3TextTransformerBlock: Module {
    fileprivate let usesSlidingWindow: Bool

    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: Mistral3SelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: Mistral3FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: Mistral3TextConfiguration, usesSlidingWindow: Bool) {
        self.usesSlidingWindow = usesSlidingWindow
        self._selfAttention.wrappedValue = Mistral3SelfAttention(config)
        self._feedForward.wrappedValue = Mistral3FeedForward(config)
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
        attentionScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = hiddenStates + selfAttention(
            inputLayerNorm(hiddenStates),
            attentionScale: attentionScale,
            mask: mask,
            cache: cache
        )
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class Mistral3Backbone: Module {
    private let schedule: Mistral3LayerSchedule
    private let slidingWindow: Int?
    private let attentionScalePlan: Mistral3AttentionScalePlan

    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [Mistral3TextTransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: Mistral3TextConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        let schedule = Mistral3LayerSchedule(config)
        self.schedule = schedule
        self.slidingWindow = config.slidingWindow
        self.attentionScalePlan = Mistral3AttentionScalePlan(config)
        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = schedule.layerTypes.enumerated().map { index, _ in
            Mistral3TextTransformerBlock(
                config,
                usesSlidingWindow: schedule.usesSlidingWindow(at: index)
            )
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ tokens: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hiddenStates = inputEmbeddings ?? tokenEmbeddings(tokens)
        let offset = cache?.first?.offset ?? 0

        let fullAttentionMask = createAttentionMask(
            h: hiddenStates,
            cache: cache?[schedule.fullAttentionMaskLayerIndex]
        )
        let slidingAttentionMask = schedule.slidingAttentionMaskLayerIndex.map { index in
            createAttentionMask(
                h: hiddenStates,
                cache: cache?[index],
                windowSize: slidingWindow
            )
        } ?? MLXFast.ScaledDotProductAttentionMaskMode.none
        let attentionScale = attentionScalePlan.values(
            start: offset,
            count: hiddenStates.dim(1),
            dtype: hiddenStates.dtype
        )

        for (layerIndex, layer) in layers.enumerated() {
            let mask = layer.usesSlidingWindow ? slidingAttentionMask : fullAttentionMask
            hiddenStates = layer(
                hiddenStates,
                attentionScale: attentionScale,
                mask: mask,
                cache: cache?[layerIndex]
            )
        }

        return finalNorm(hiddenStates)
    }
}

private struct Mistral3WeightSanitizer {
    let tieWordEmbeddings: Bool

    func callAsFunction(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = Self.unwrapLanguageModelWeights(weights)
        sanitized = sanitized.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        if tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return MLXQuantizedWeightSanitizer.sanitize(
            sanitized,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights
    }

    private static func unwrapLanguageModelWeights(
        _ weights: [String: MLXArray]
    ) -> [String: MLXArray] {
        let unflattened = ModuleParameters.unflattened(weights)
        guard let languageModel = unflattened["language_model"] else {
            return weights
        }
        return Dictionary(uniqueKeysWithValues: languageModel.flattened())
    }
}

internal final class Mistral3TextModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    fileprivate let model: Mistral3Backbone
    private let config: Mistral3TextConfiguration

    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    internal init(_ config: Mistral3TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.model = Mistral3Backbone(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, inputEmbeddings: nil)
    }

    internal func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputEmbeddings: MLXArray?
    ) -> MLXArray {
        logits(from: model(inputs, cache: cache, inputEmbeddings: inputEmbeddings))
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
        Mistral3WeightSanitizer(tieWordEmbeddings: config.tieWordEmbeddings)(weights)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { layer in
            if layer.usesSlidingWindow, let slidingWindow = config.slidingWindow {
                RotatingKVCache(maxSize: slidingWindow)
            } else {
                KVCacheSimple()
            }
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

internal struct Mistral3TextConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var kvHeads: Int
    var ropeTheta: Float
    var ropeParameters: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var layerTypes: [String]
    var slidingWindow: Int?

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    internal init(from decoder: Decoder) throws {
        let topLevelContainer = try decoder.container(keyedBy: CodingKeys.self)
        let vlmContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
        let container = if vlmContainer.contains(.textConfig) {
            try vlmContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            topLevelContainer
        }

        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "ministral3",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: hiddenLayers,
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeParameters: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeParameters
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? (try topLevelContainer.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings))
                ?? false,
            layerTypes: try container.decodeIfPresent([String].self, forKey: .layerTypes)
                ?? Array(repeating: Mistral3LayerSchedule.fullAttention, count: hiddenLayers),
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        )
    }

    internal init(
        modelType: String = "ministral3",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        headDimensions: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        kvHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeParameters: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        layerTypes: [String]? = nil,
        slidingWindow: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDimensions = headDimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeParameters = ropeParameters
        self.tieWordEmbeddings = tieWordEmbeddings
        self.layerTypes = layerTypes
            ?? Array(repeating: Mistral3LayerSchedule.fullAttention, count: hiddenLayers)
        self.slidingWindow = slidingWindow
    }
}

extension Mistral3TextModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
