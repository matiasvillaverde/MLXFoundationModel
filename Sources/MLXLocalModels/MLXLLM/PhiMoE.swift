import Foundation
import MLX
import MLXFast
import MLXNN

internal struct PhiMoEAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: PhiMoEConfiguration) {
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

internal struct PhiMoERouterPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let expertsPerToken: Int

    internal init(_ config: PhiMoEConfiguration) {
        precondition(config.numLocalExperts > 0, "num_local_experts must be positive")
        precondition(config.numExpertsPerToken > 0, "num_experts_per_tok must be positive")
        precondition(
            config.numExpertsPerToken <= config.numLocalExperts,
            "num_experts_per_tok cannot exceed num_local_experts"
        )

        self.expertCount = config.numLocalExperts
        self.expertsPerToken = config.numExpertsPerToken
    }
}

internal struct PhiMoERotaryPlan: Equatable, Sendable {
    internal enum Kind: Equatable, Sendable {
        case rope(scale: Float)
        case longRoPE(
            shortFactor: [Float],
            longFactor: [Float],
            shortMScale: Float?,
            longMScale: Float?
        )
    }

    internal let dimensions: Int
    internal let base: Float
    internal let maxPositionEmbeddings: Int
    internal let originalMaxPositionEmbeddings: Int
    internal let kind: Kind

    internal init(_ config: PhiMoEConfiguration, layout: PhiMoEAttentionLayout) {
        precondition(config.maxPositionEmbeddings > 0, "max_position_embeddings must be positive")
        precondition(
            config.originalMaxPositionEmbeddings > 0,
            "original_max_position_embeddings must be positive"
        )

        self.dimensions = layout.headSize
        self.base = config.ropeTheta
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = config.originalMaxPositionEmbeddings

        if let scaling = config.ropeScaling,
           scaling.usesLongRoPE || scaling.longFactor != nil || scaling.shortFactor != nil {
            self.kind = .longRoPE(
                shortFactor: scaling.shortFactor ?? [1],
                longFactor: scaling.longFactor ?? [1],
                shortMScale: scaling.shortMScale,
                longMScale: scaling.longMScale
            )
        } else if config.ropeScaling?.type == "linear",
                  let factor = config.ropeScaling?.factor,
                  factor > 0 {
            self.kind = .rope(scale: 1 / factor)
        } else {
            self.kind = .rope(scale: 1)
        }
    }
}

private enum PhiMoERotaryEmbedding {
    case rope(RoPE)
    case longRoPE(SuScaledRoPE)

    init(_ plan: PhiMoERotaryPlan) {
        switch plan.kind {
        case .rope(let scale):
            self = .rope(
                RoPE(
                    dimensions: plan.dimensions,
                    traditional: false,
                    base: plan.base,
                    scale: scale
                )
            )
        case .longRoPE(let shortFactor, let longFactor, let shortMScale, let longMScale):
            self = .longRoPE(
                SuScaledRoPE(
                    dimensions: plan.dimensions,
                    base: plan.base,
                    maxPositionEmbeddings: plan.maxPositionEmbeddings,
                    originalMaxPositionEmbeddings: plan.originalMaxPositionEmbeddings,
                    shortFactor: shortFactor,
                    longFactor: longFactor,
                    shortMScale: shortMScale,
                    longMScale: longMScale
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

private final class PhiMoEAttention: Module {
    private let layout: PhiMoEAttentionLayout
    private let rotaryEmbedding: PhiMoERotaryEmbedding

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: PhiMoEConfiguration) {
        let layout = PhiMoEAttentionLayout(config)
        self.layout = layout
        self.rotaryEmbedding = PhiMoERotaryEmbedding(
            PhiMoERotaryPlan(config, layout: layout)
        )

        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: true
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: true
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: true
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: true
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

private final class PhiMoESparseBlock: Module {
    private let routerPlan: PhiMoERouterPlan

    @ModuleInfo(key: "gate") private var gate: Linear
    @ModuleInfo(key: "switch_mlp") private var experts: SwitchGLU

    init(_ config: PhiMoEConfiguration) {
        let routerPlan = PhiMoERouterPlan(config)
        self.routerPlan = routerPlan
        self._gate.wrappedValue = Linear(
            config.hiddenSize,
            routerPlan.expertCount,
            bias: false
        )
        self._experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: routerPlan.expertCount
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let routerLogits = gate(hiddenStates)
        let expertIndices = stopGradient(
            argPartition(
                -routerLogits,
                kth: routerPlan.expertsPerToken - 1,
                axis: -1
            )[.ellipsis, ..<routerPlan.expertsPerToken]
        )
        let routingWeights = softmax(
            takeAlong(routerLogits, expertIndices, axis: -1),
            axis: -1,
            precise: true
        )

        return (experts(hiddenStates, expertIndices) * routingWeights[.ellipsis, .newAxis])
            .sum(axis: -2)
    }
}

private final class PhiMoEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: PhiMoEAttention
    @ModuleInfo(key: "block_sparse_moe") private var sparseBlock: PhiMoESparseBlock
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: LayerNorm

    init(_ config: PhiMoEConfiguration) {
        self._selfAttention.wrappedValue = PhiMoEAttention(config)
        self._sparseBlock.wrappedValue = PhiMoESparseBlock(config)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = LayerNorm(
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
        return afterAttention + sparseBlock(postAttentionLayerNorm(afterAttention))
    }
}

private final class PhiMoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [PhiMoEDecoderLayer]
    @ModuleInfo(key: "norm") private var finalNorm: LayerNorm

    init(_ config: PhiMoEConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            PhiMoEDecoderLayer(config)
        }
        self._finalNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
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

internal final class PhiMoEModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") fileprivate var model: PhiMoEBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    private let configuration: PhiMoEConfiguration

    public init(_ config: PhiMoEConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self._model.wrappedValue = PhiMoEBackbone(config)
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize,
            config.vocabularySize,
            bias: true
        )
    }

    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(tokens, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: lmHead(hiddenStates), state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        PhiMoEWeightSanitizer.sanitize(
            weights,
            layerCount: configuration.hiddenLayers,
            expertCount: configuration.numLocalExperts
        )
    }
}

private enum PhiMoEWeightSanitizer {
    private static let projectionNameMap = [
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj")
    ]
    private static let tensorSuffixes = ["weight", "scales", "biases"]

    static func sanitize(
        _ weights: [String: MLXArray],
        layerCount: Int,
        expertCount: Int
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }

        for layerIndex in 0 ..< layerCount {
            let layerPrefix = "model.layers.\(layerIndex).block_sparse_moe"
            for (sourceProjection, targetProjection) in projectionNameMap {
                for suffix in tensorSuffixes {
                    let sourceKeys = (0 ..< expertCount).map { expertIndex in
                        "\(layerPrefix).experts.\(expertIndex).\(sourceProjection).\(suffix)"
                    }
                    guard sourceKeys.allSatisfy({ sanitized[$0] != nil }) else { continue }

                    let stackedTensors = sourceKeys.compactMap { sanitized.removeValue(forKey: $0) }
                    sanitized["\(layerPrefix).switch_mlp.\(targetProjection).\(suffix)"] =
                        stacked(stackedTensors)
                }
            }
        }

        return sanitized
    }
}

internal struct PhiMoEConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var originalMaxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeScaling: RopeScalingWithFactorArrays?
    var numLocalExperts: Int
    var numExpertsPerToken: Int
    var ropeTheta: Float

    internal init(
        modelType: String = "phimoe",
        vocabularySize: Int = 32_064,
        hiddenSize: Int = 4_096,
        intermediateSize: Int = 6_400,
        hiddenLayers: Int = 32,
        attentionHeads: Int = 32,
        kvHeads: Int = 8,
        maxPositionEmbeddings: Int = 131_072,
        originalMaxPositionEmbeddings: Int = 4_096,
        rmsNormEps: Float = 1e-6,
        ropeScaling: RopeScalingWithFactorArrays? = nil,
        numLocalExperts: Int = 16,
        numExpertsPerToken: Int = 2,
        ropeTheta: Float = 10_000
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeScaling = ropeScaling
        self.numLocalExperts = numLocalExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.ropeTheta = ropeTheta
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeScaling = "rope_scaling"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case ropeTheta = "rope_theta"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "phimoe",
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 32_064,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4_096,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 6_400,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32,
            attentionHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .attentionHeads
            ) ?? 32,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            originalMaxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .originalMaxPositionEmbeddings
            ) ?? 4_096,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            ropeScaling: try container.decodeIfPresent(
                RopeScalingWithFactorArrays.self,
                forKey: .ropeScaling
            ),
            numLocalExperts: try container.decodeIfPresent(
                Int.self,
                forKey: .numLocalExperts
            ) ?? 16,
            numExpertsPerToken: try container.decodeIfPresent(
                Int.self,
                forKey: .numExpertsPerToken
            ) ?? 2,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        )
    }
}

extension PhiMoEModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
