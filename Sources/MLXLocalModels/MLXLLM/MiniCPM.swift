import Foundation
import MLX
import MLXFast
import MLXNN

internal struct MiniCPMAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: MiniCPMConfiguration) {
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
        self.headSize = config.hiddenSize / config.attentionHeads
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

internal struct MiniCPMScalePlan: Equatable, Sendable {
    internal let embeddingScale: Float
    internal let residualScale: Float
    internal let logitDivisor: Float

    internal init(_ config: MiniCPMConfiguration) {
        precondition(config.hiddenLayers > 0, "num_hidden_layers must be positive")
        precondition(config.dimModelBase > 0, "dim_model_base must be positive")

        self.embeddingScale = config.scaleEmb
        self.residualScale = config.scaleDepth / Float(config.hiddenLayers).squareRoot()
        self.logitDivisor = Float(config.hiddenSize) / Float(config.dimModelBase)
    }
}

private final class MiniCPMSelfAttention: Module {
    private let layout: MiniCPMAttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: MiniCPMConfiguration) {
        let layout = MiniCPMAttentionLayout(config)
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
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
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

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
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

private final class MiniCPMFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: MiniCPMConfiguration) {
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

private final class MiniCPMTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: MiniCPMSelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: MiniCPMFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    private let residualScale: Float

    init(_ config: MiniCPMConfiguration, scalePlan: MiniCPMScalePlan) {
        self.residualScale = scalePlan.residualScale
        self._selfAttention.wrappedValue = MiniCPMSelfAttention(config)
        self._feedForward.wrappedValue = MiniCPMFeedForward(config)
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
            * residualScale
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
            * residualScale
    }
}

private final class MiniCPMBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [MiniCPMTransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    private let scalePlan: MiniCPMScalePlan

    init(_ config: MiniCPMConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        let scalePlan = MiniCPMScalePlan(config)
        self.scalePlan = scalePlan
        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            MiniCPMTransformerBlock(config, scalePlan: scalePlan)
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        if scalePlan.embeddingScale != 1 {
            hiddenStates = hiddenStates * scalePlan.embeddingScale
        }

        let mask = createAttentionMask(h: hiddenStates, cache: cache)
        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                mask: mask,
                cache: cache?[layerIndex]
            )
        }

        return finalNorm(hiddenStates)
    }
}

internal final class MiniCPMModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") fileprivate var model: MiniCPMBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    private let scalePlan: MiniCPMScalePlan
    private let tieWordEmbeddings: Bool

    internal init(_ config: MiniCPMConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.scalePlan = MiniCPMScalePlan(config)
        self.tieWordEmbeddings = config.tieWordEmbeddings
        self._model.wrappedValue = MiniCPMBackbone(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
        }
    }

    internal func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
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

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            if key.contains("self_attn.rotary_emb.inv_freq") { return false }
            if tieWordEmbeddings, key == "lm_head.weight" { return false }
            return true
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        let scaledHiddenStates = scalePlan.logitDivisor == 1
            ? hiddenStates
            : hiddenStates / scalePlan.logitDivisor
        if let lmHead {
            return lmHead(scaledHiddenStates)
        }
        return model.tokenEmbeddings.asLinear(scaledHiddenStates)
    }
}

internal struct MiniCPMConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
    var dimModelBase: Int
    var scaleDepth: Float
    var scaleEmb: Float
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "minicpm",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        maxPositionEmbeddings: Int,
        dimModelBase: Int? = nil,
        scaleDepth: Float = 1,
        scaleEmb: Float = 1,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.dimModelBase = dimModelBase ?? hiddenSize
        self.scaleDepth = scaleDepth
        self.scaleEmb = scaleEmb
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case dimModelBase = "dim_model_base"
        case scaleDepth = "scale_depth"
        case scaleEmb = "scale_emb"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "minicpm",
            hiddenSize: hiddenSize,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            maxPositionEmbeddings: try container.decode(Int.self, forKey: .maxPositionEmbeddings),
            dimModelBase: try container.decodeIfPresent(Int.self, forKey: .dimModelBase)
                ?? hiddenSize,
            scaleDepth: try container.decodeIfPresent(Float.self, forKey: .scaleDepth) ?? 1,
            scaleEmb: try container.decodeIfPresent(Float.self, forKey: .scaleEmb) ?? 1,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

extension MiniCPMModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
