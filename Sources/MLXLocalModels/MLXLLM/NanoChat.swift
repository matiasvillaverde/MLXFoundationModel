import Foundation
import MLX
import MLXFast
import MLXNN

internal struct NanoChatAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: NanoChatConfiguration) {
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

        let headSize = config.hiddenSize / config.attentionHeads
        precondition(headSize.isMultiple(of: 2), "head_dim must be even for RoPE")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = headSize
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

internal struct NanoChatRotaryPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let halfDimensions: Int
    internal let theta: Float

    internal init(layout: NanoChatAttentionLayout, theta: Float) {
        precondition(theta > 0, "rope_theta must be positive")
        self.dimensions = layout.headSize
        self.halfDimensions = layout.headSize / 2
        self.theta = theta
    }

    internal func frequencies() -> MLXArray {
        let indices = MLXArray(Array(0 ..< halfDimensions)).asType(.float32)
        let scale = Float(log(Double(theta)) / Double(halfDimensions))
        return -MLX.exp(indices * scale)
    }
}

internal struct NanoChatLogitSoftcap: Equatable, Sendable {
    internal let cap: Float

    internal init(_ cap: Float) {
        self.cap = cap
    }

    internal func apply(to logits: MLXArray) -> MLXArray {
        guard cap > 0 else {
            return logits
        }
        let scale = MLXArray(cap)
        return scale * tanh(logits / scale)
    }
}

private func nanoChatRMSNorm(_ input: MLXArray, eps: Float) -> MLXArray {
    let meanSquares = mean(input.square(), axis: -1, keepDims: true)
    return input * (meanSquares + eps).rsqrt()
}

private final class NanoChatRotaryEmbedding: Module, OffsetLayer, ArrayOffsetLayer {
    private let dimensions: Int
    private let freqs: MLXArray

    init(_ plan: NanoChatRotaryPlan) {
        self.dimensions = plan.dimensions
        self.freqs = plan.frequencies()
    }

    func callAsFunction(_ input: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            input,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1,
            offset: offset,
            freqs: freqs
        )
    }

    func callAsFunction(_ input: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            input,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1,
            offset: offset,
            freqs: freqs
        )
    }
}

private final class NanoChatSelfAttention: Module {
    private let layout: NanoChatAttentionLayout
    private let normalizationEps: Float
    private let rope: RoPELayer

    @ModuleInfo(key: "c_q") private var queryProjection: Linear
    @ModuleInfo(key: "c_k") private var keyProjection: Linear
    @ModuleInfo(key: "c_v") private var valueProjection: Linear
    @ModuleInfo(key: "c_proj") private var outputProjection: Linear

    init(_ config: NanoChatConfiguration) {
        let layout = NanoChatAttentionLayout(config)
        self.layout = layout
        self.normalizationEps = config.rmsNormEps
        self.rope = NanoChatRotaryEmbedding(
            NanoChatRotaryPlan(layout: layout, theta: config.ropeTheta)
        )

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

        queries = nanoChatRMSNorm(
            applyRotaryPosition(rope, to: queries, cache: cache),
            eps: normalizationEps
        )
        keys = nanoChatRMSNorm(
            applyRotaryPosition(rope, to: keys, cache: cache),
            eps: normalizationEps
        )

        let attentionOutput = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, layout.queryProjectionSize)

        return outputProjection(attentionOutput)
    }
}

private final class NanoChatFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") private var inputProjection: Linear
    @ModuleInfo(key: "c_proj") private var outputProjection: Linear

    init(_ config: NanoChatConfiguration) {
        self._inputProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let activated = relu(inputProjection(hiddenStates))
        return outputProjection(activated * activated)
    }
}

private final class NanoChatBlock: Module {
    private let normalizationEps: Float

    @ModuleInfo(key: "attn") fileprivate var attention: NanoChatSelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: NanoChatFeedForward

    init(_ config: NanoChatConfiguration) {
        self.normalizationEps = config.rmsNormEps
        self._attention.wrappedValue = NanoChatSelfAttention(config)
        self._feedForward.wrappedValue = NanoChatFeedForward(config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = hiddenStates + attention(
            nanoChatRMSNorm(hiddenStates, eps: normalizationEps),
            mask: mask,
            cache: cache
        )
        return afterAttention + feedForward(nanoChatRMSNorm(afterAttention, eps: normalizationEps))
    }
}

private final class NanoChatBackbone: Module {
    private let normalizationEps: Float

    @ModuleInfo(key: "wte") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "h") fileprivate var layers: [NanoChatBlock]

    init(_ config: NanoChatConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")
        self.normalizationEps = config.rmsNormEps
        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            NanoChatBlock(config)
        }
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = nanoChatRMSNorm(tokenEmbeddings(tokens), eps: normalizationEps)
        let mask = createAttentionMask(h: hiddenStates, cache: cache?.first)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return nanoChatRMSNorm(hiddenStates, eps: normalizationEps)
    }
}

internal final class NanoChatModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let modelType: String

    @ModuleInfo(key: "transformer") private var transformer: NanoChatBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    private let config: NanoChatConfiguration
    private let logitSoftcap: NanoChatLogitSoftcap

    internal init(_ config: NanoChatConfiguration) {
        self.config = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.logitSoftcap = NanoChatLogitSoftcap(config.logitsSoftcap)
        self._transformer.wrappedValue = NanoChatBackbone(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
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

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { _ in KVCacheSimple() }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        logitSoftcap.apply(to: lmHead(hiddenStates))
    }
}

internal struct NanoChatConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var attentionHeads: Int
    internal var kvHeads: Int
    internal var vocabularySize: Int
    internal var maxPositionEmbeddings: Int
    internal var intermediateSize: Int
    internal var ropeTheta: Float
    internal var rmsNormEps: Float
    internal var logitsSoftcap: Float

    internal init(
        modelType: String = "nanochat",
        hiddenSize: Int,
        hiddenLayers: Int,
        attentionHeads: Int,
        kvHeads: Int? = nil,
        vocabularySize: Int,
        maxPositionEmbeddings: Int,
        intermediateSize: Int,
        ropeTheta: Float = 10_000,
        rmsNormEps: Float = 1e-5,
        logitsSoftcap: Float = 15
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads ?? attentionHeads
        self.vocabularySize = vocabularySize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.intermediateSize = intermediateSize
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.logitsSoftcap = logitsSoftcap
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case logitsSoftcap = "logits_softcap"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "nanochat",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            attentionHeads: attentionHeads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            maxPositionEmbeddings: try container.decode(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            logitsSoftcap: try container.decodeIfPresent(Float.self, forKey: .logitsSoftcap)
                ?? 15
        )
    }
}

extension NanoChatModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        transformer.layers.map { ($0.attention, ["c_q", "c_v"]) }
    }
}
