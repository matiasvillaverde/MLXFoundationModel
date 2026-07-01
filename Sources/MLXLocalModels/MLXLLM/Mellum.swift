import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

internal enum MellumLayerKind: String, Codable, Sendable, Equatable {
    case full = "full_attention"
    case sliding = "sliding_attention"
}

internal struct MellumConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var numExperts: Int
    var numExpertsPerToken: Int
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDimensions: Int
    var tieWordEmbeddings: Bool
    var maxPositionEmbeddings: Int
    var normTopKProbabilities: Bool
    var slidingWindow: Int
    var layerTypes: [MellumLayerKind]
    var ropeParameters: [String: [String: StringOrNumber]]

    internal init(
        modelType: String = "mellum",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        numExperts: Int,
        numExpertsPerToken: Int,
        moeIntermediateSize: Int,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int,
        kvHeads: Int,
        headDimensions: Int,
        tieWordEmbeddings: Bool = false,
        maxPositionEmbeddings: Int = 131_072,
        normTopKProbabilities: Bool = true,
        slidingWindow: Int = 1_024,
        layerTypes: [MellumLayerKind]? = nil,
        ropeParameters: [String: [String: StringOrNumber]]? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.numExperts = numExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.moeIntermediateSize = moeIntermediateSize
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.headDimensions = headDimensions
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normTopKProbabilities = normTopKProbabilities
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes ?? Array(repeating: .full, count: hiddenLayers)
        self.ropeParameters = ropeParameters ?? [
            MellumLayerKind.full.rawValue: [
                "rope_type": .string("default"),
                "rope_theta": .int(10_000)
            ],
            MellumLayerKind.sliding.rawValue: [
                "rope_type": .string("default"),
                "rope_theta": .int(10_000)
            ]
        ]
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normTopKProbabilities = "norm_topk_prob"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        let layerTypes = try container.decode([MellumLayerKind].self, forKey: .layerTypes)
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription: "Mellum layer_types count must match num_hidden_layers"
            )
        }

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "mellum",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: hiddenLayers,
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            numExperts: try container.decode(Int.self, forKey: .numExperts),
            numExpertsPerToken: try container.decode(Int.self, forKey: .numExpertsPerToken),
            moeIntermediateSize: try container.decode(Int.self, forKey: .moeIntermediateSize),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decode(Int.self, forKey: .kvHeads),
            headDimensions: try container.decode(Int.self, forKey: .headDimensions),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            maxPositionEmbeddings: try container.decode(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            normTopKProbabilities: try container.decode(
                Bool.self,
                forKey: .normTopKProbabilities
            ),
            slidingWindow: try container.decode(Int.self, forKey: .slidingWindow),
            layerTypes: layerTypes,
            ropeParameters: try container.decode(
                [String: [String: StringOrNumber]].self,
                forKey: .ropeParameters
            )
        )
    }
}

internal struct MellumLayerPlan: Equatable, Sendable {
    let kinds: [MellumLayerKind]
    let firstFullLayer: Int?
    let firstSlidingLayer: Int?
    let slidingWindow: Int

    init(_ config: MellumConfiguration) {
        precondition(config.hiddenLayers > 0, "Mellum must have at least one layer")
        precondition(
            config.layerTypes.count == config.hiddenLayers,
            "Mellum layer_types must match hidden layer count"
        )
        precondition(config.slidingWindow > 0, "Mellum sliding window must be positive")

        self.kinds = config.layerTypes
        self.firstFullLayer = kinds.firstIndex(of: .full)
        self.firstSlidingLayer = kinds.firstIndex(of: .sliding)
        self.slidingWindow = config.slidingWindow
    }

    func kind(layerIndex: Int) -> MellumLayerKind {
        kinds[layerIndex]
    }
}

internal struct MellumAttentionLayout: Equatable, Sendable {
    let hiddenSize: Int
    let queryHeads: Int
    let keyValueHeads: Int
    let headDimensions: Int
    let queryProjectionSize: Int
    let keyValueProjectionSize: Int
    let attentionScale: Float

    init(_ config: MellumConfiguration) {
        precondition(config.hiddenSize > 0, "Mellum hidden size must be positive")
        precondition(config.attentionHeads > 0, "Mellum attention heads must be positive")
        precondition(config.kvHeads > 0, "Mellum KV heads must be positive")
        precondition(config.headDimensions > 0, "Mellum head_dim must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "Mellum attention heads must group KV heads"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = config.headDimensions
        self.queryProjectionSize = queryHeads * headDimensions
        self.keyValueProjectionSize = keyValueHeads * headDimensions
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }
}

private struct MellumRoPEPlan: Equatable, Sendable {
    let base: Float
    let scalingConfig: [String: StringOrNumber]?

    init(_ config: MellumConfiguration, layerKind: MellumLayerKind) {
        let parameters = config.ropeParameters[layerKind.rawValue] ?? [:]
        self.base = parameters["rope_theta"]?.asFloat() ?? 10_000

        let ropeType: String = {
            if let value = parameters["rope_type"] ?? parameters["type"],
               case .string(let string) = value {
                return string
            }
            return "default"
        }()

        if ropeType == "default" || ropeType == "linear" {
            self.scalingConfig = nil
        } else {
            var scaling = parameters
            scaling["type"] = .string(ropeType)
            self.scalingConfig = scaling
        }
    }
}

private enum MellumExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case up = "up_proj"
    case down = "down_proj"
}

// MARK: - Model Components

private final class MellumSelfAttention: Module {
    let layout: MellumAttentionLayout

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") private var keyNorm: RMSNorm

    private let rope: RoPELayer

    init(_ config: MellumConfiguration, layerKind: MellumLayerKind) {
        self.layout = MellumAttentionLayout(config)

        _queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        _keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: false
        )
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: layout.headDimensions,
            eps: config.rmsNormEps
        )
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: layout.headDimensions,
            eps: config.rmsNormEps
        )

        let ropePlan = MellumRoPEPlan(config, layerKind: layerKind)
        self.rope = initializeRope(
            dims: layout.headDimensions,
            base: ropePlan.base,
            traditional: false,
            scalingConfig: ropePlan.scalingConfig,
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
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headDimensions)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
        let values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        queries = queryNorm(queries).transposed(0, 2, 1, 3)
        keys = keyNorm(keys).transposed(0, 2, 1, 3)

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

internal struct MellumRouterPlan: Equatable, Sendable {
    let expertCount: Int
    let selectedExpertCount: Int
    let normalizesTopKProbabilities: Bool

    init(_ config: MellumConfiguration) {
        precondition(config.numExperts > 0, "Mellum num_experts must be positive")
        precondition(
            config.numExpertsPerToken > 0,
            "Mellum num_experts_per_tok must be positive"
        )
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "Mellum num_experts_per_tok must be <= num_experts"
        )

        self.expertCount = config.numExperts
        self.selectedExpertCount = config.numExpertsPerToken
        self.normalizesTopKProbabilities = config.normTopKProbabilities
    }
}

private final class MellumSparseMoEBlock: Module, UnaryLayer {
    let routerPlan: MellumRouterPlan

    @ModuleInfo(key: "gate") private var gate: Linear
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU

    init(_ config: MellumConfiguration) {
        self.routerPlan = MellumRouterPlan(config)
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let probabilities = softmax(gate(hiddenStates), axis: -1, precise: true)
        let topK = routerPlan.selectedExpertCount
        let threshold = routerPlan.expertCount - topK
        let expertIndices = argPartition(probabilities, kth: threshold, axis: -1)[
            .ellipsis, threshold...
        ]
        var expertScores = takeAlong(probabilities, expertIndices, axis: -1)
        if routerPlan.normalizesTopKProbabilities {
            expertScores = expertScores / expertScores.sum(axis: -1, keepDims: true)
        }

        let expertOutputs = switchMLP(hiddenStates, expertIndices)
        return (expertOutputs * expertScores.asType(hiddenStates.dtype)[.ellipsis, .newAxis])
            .sum(axis: -2)
    }
}

private final class MellumDecoderLayer: Module {
    let layerKind: MellumLayerKind

    @ModuleInfo(key: "self_attn") var selfAttention: MellumSelfAttention
    @ModuleInfo(key: "mlp") private var feedForward: MellumSparseMoEBlock
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: MellumConfiguration, layerIndex: Int, layerPlan: MellumLayerPlan) {
        self.layerKind = layerPlan.kind(layerIndex: layerIndex)
        _selfAttention.wrappedValue = MellumSelfAttention(config, layerKind: layerKind)
        _feedForward.wrappedValue = MellumSparseMoEBlock(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
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

private final class MellumBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [MellumDecoderLayer]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    let layerPlan: MellumLayerPlan

    init(_ config: MellumConfiguration) {
        precondition(config.vocabularySize > 0, "Mellum vocab size must be positive")
        let plan = MellumLayerPlan(config)
        self.layerPlan = plan

        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            MellumDecoderLayer(config, layerIndex: layerIndex, layerPlan: plan)
        }
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)

        let fullMask = layerPlan.firstFullLayer.map { layerIndex in
            createAttentionMask(h: hiddenStates, cache: cache?[layerIndex])
        } ?? .none
        let slidingMask = layerPlan.firstSlidingLayer.map { layerIndex in
            createAttentionMask(
                h: hiddenStates,
                cache: cache?[layerIndex],
                windowSize: layerPlan.slidingWindow
            )
        }

        for (layerIndex, layer) in layers.enumerated() {
            let mask = layer.layerKind == .sliding ? (slidingMask ?? fullMask) : fullMask
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

private struct MellumWeightSanitizer {
    let config: MellumConfiguration

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }

        packExperts(in: &sanitized)
        return sanitized
    }

    private func packExperts(in weights: inout [String: MLXArray]) {
        guard config.numExperts > 0 else { return }

        for layerIndex in 0 ..< config.hiddenLayers {
            let mlpPrefix = "model.layers.\(layerIndex).mlp"
            let expertsPrefix = "\(mlpPrefix).experts"
            guard weights["\(expertsPrefix).0.up_proj.weight"] != nil else {
                continue
            }

            for projection in MellumExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let keys = (0 ..< config.numExperts).map { expertIndex in
                        "\(expertsPrefix).\(expertIndex).\(projection.rawValue).\(tensorName)"
                    }
                    guard keys.allSatisfy({ weights[$0] != nil }) else {
                        continue
                    }
                    let tensors = keys.compactMap { weights.removeValue(forKey: $0) }
                    weights["\(mlpPrefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        MLX.stacked(tensors)
                }
            }
        }
    }
}

internal final class MellumModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let modelType: String
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let config: MellumConfiguration
    @ModuleInfo(key: "model") fileprivate var model: MellumBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: MellumConfiguration) {
        self.config = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _model.wrappedValue = MellumBackbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
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
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        MellumWeightSanitizer(config: config).sanitize(weights)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        MellumLayerPlan(config).kinds.map { kind in
            switch kind {
            case .full:
                KVCacheSimple()
            case .sliding:
                RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
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

extension MellumModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
