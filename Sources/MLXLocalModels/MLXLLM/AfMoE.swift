import Foundation
import MLX
import MLXNN

// MARK: - Configuration

internal struct AfMoEConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var numExperts: Int
    var numExpertsPerToken: Int
    var numSharedExperts: Int
    var numDenseLayers: Int
    var routeNorm: Bool
    var routeScale: Float
    var scoreFunc: String
    var nGroup: Int
    var topkGroup: Int
    var layerTypes: [String]
    var slidingWindow: Int
    var mupEnabled: Bool

    internal init(
        modelType: String = "afmoe",
        vocabularySize: Int = 200_192,
        hiddenSize: Int = 2_048,
        intermediateSize: Int = 6_144,
        moeIntermediateSize: Int = 1_024,
        hiddenLayers: Int = 32,
        attentionHeads: Int = 32,
        kvHeads: Int = 4,
        headDim: Int = 64,
        maxPositionEmbeddings: Int = 131_072,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        numExperts: Int = 128,
        numExpertsPerToken: Int = 8,
        numSharedExperts: Int = 1,
        numDenseLayers: Int = 2,
        routeNorm: Bool = true,
        routeScale: Float = 2.826,
        scoreFunc: String = "sigmoid",
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        layerTypes: [String]? = nil,
        slidingWindow: Int = 2_048,
        mupEnabled: Bool = true
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.numExperts = numExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.numSharedExperts = numSharedExperts
        self.numDenseLayers = numDenseLayers
        self.routeNorm = routeNorm
        self.routeScale = routeScale
        self.scoreFunc = scoreFunc
        self.nGroup = nGroup
        self.topkGroup = topkGroup ?? nGroup
        self.layerTypes = layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
        self.slidingWindow = slidingWindow
        self.mupEnabled = mupEnabled
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case numDenseLayers = "num_dense_layers"
        case routeNorm = "route_norm"
        case routeScale = "route_scale"
        case scoreFunc = "score_func"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case mupEnabled = "mup_enabled"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        let nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "afmoe",
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 200_192,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? 6_144,
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ) ?? 1_024,
            hiddenLayers: hiddenLayers,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            numExperts: try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128,
            numExpertsPerToken: try container.decodeIfPresent(
                Int.self,
                forKey: .numExpertsPerToken
            ) ?? 8,
            numSharedExperts: try container.decodeIfPresent(
                Int.self,
                forKey: .numSharedExperts
            ) ?? 1,
            numDenseLayers: try container.decodeIfPresent(Int.self, forKey: .numDenseLayers) ?? 2,
            routeNorm: try container.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true,
            routeScale: try container.decodeIfPresent(Float.self, forKey: .routeScale) ?? 2.826,
            scoreFunc: try container.decodeIfPresent(String.self, forKey: .scoreFunc)
                ?? "sigmoid",
            nGroup: nGroup,
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? nGroup,
            layerTypes: try container.decodeIfPresent([String].self, forKey: .layerTypes),
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
                ?? 2_048,
            mupEnabled: try container.decodeIfPresent(Bool.self, forKey: .mupEnabled) ?? true
        )
    }
}

// MARK: - Plans

internal enum AfMoEAttentionKind: String, Equatable, Sendable {
    case full = "full_attention"
    case sliding = "sliding_attention"

    internal init(_ value: String) {
        self = AfMoEAttentionKind(rawValue: value) ?? .full
    }
}

internal struct AfMoEAttentionLayout: Equatable, Sendable {
    internal let kind: AfMoEAttentionKind
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let attentionScale: Float
    internal let usesRotaryPosition: Bool

    internal init(_ config: AfMoEConfiguration, kind: AfMoEAttentionKind) {
        precondition(config.hiddenSize > 0, "AfMoE hidden size must be positive")
        precondition(config.attentionHeads > 0, "AfMoE attention heads must be positive")
        precondition(config.kvHeads > 0, "AfMoE KV heads must be positive")
        precondition(config.headDim > 0, "AfMoE head dimension must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "AfMoE attention heads must group KV heads"
        )

        self.kind = kind
        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = config.headDim
        self.attentionScale = pow(Float(config.headDim), -0.5)
        self.usesRotaryPosition = kind == .sliding
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
}

internal struct AfMoELayerPlan: Equatable, Sendable {
    internal let attentionKinds: [AfMoEAttentionKind]
    internal let denseLayerCount: Int
    internal let slidingWindow: Int

    internal init(_ config: AfMoEConfiguration) {
        precondition(config.hiddenLayers > 0, "AfMoE must have at least one layer")
        precondition(
            config.layerTypes.count == config.hiddenLayers,
            "AfMoE layer_types must match hidden layer count"
        )
        precondition(config.numDenseLayers >= 0, "AfMoE dense-layer count is negative")
        precondition(
            config.numDenseLayers <= config.hiddenLayers,
            "AfMoE dense-layer count exceeds layer count"
        )
        precondition(config.slidingWindow > 0, "AfMoE sliding window must be positive")

        self.attentionKinds = config.layerTypes.map(AfMoEAttentionKind.init)
        self.denseLayerCount = config.numDenseLayers
        self.slidingWindow = config.slidingWindow
    }

    internal var layerCount: Int {
        attentionKinds.count
    }

    internal var firstFullLayer: Int {
        attentionKinds.firstIndex(of: .full) ?? 0
    }

    internal var firstSlidingLayer: Int? {
        attentionKinds.firstIndex(of: .sliding)
    }

    internal func attentionKind(layerIndex: Int) -> AfMoEAttentionKind {
        attentionKinds[layerIndex]
    }

    internal func usesSparseExperts(layerIndex: Int) -> Bool {
        layerIndex >= denseLayerCount
    }
}

internal enum AfMoEScoreFunction: String, Equatable, Sendable {
    case sigmoid
    case softmax

    internal init(_ rawValue: String) {
        self = AfMoEScoreFunction(rawValue: rawValue) ?? .sigmoid
    }

    internal func scores(from logits: MLXArray) -> MLXArray {
        switch self {
        case .sigmoid:
            MLX.sigmoid(logits.asType(.float32))
        case .softmax:
            MLX.softmax(logits.asType(.float32), axis: -1)
        }
    }
}

internal struct AfMoERoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizesSelectedProbabilities: Bool
    internal let routedScalingFactor: Float
    internal let scoreFunction: AfMoEScoreFunction

    internal init(_ config: AfMoEConfiguration) {
        let keptGroupCount = min(config.topkGroup, config.nGroup)

        precondition(config.numExperts > 0, "AfMoE expert count must be positive")
        precondition(config.numExpertsPerToken > 0, "AfMoE top-k must be positive")
        precondition(config.nGroup > 0, "AfMoE group count must be positive")
        precondition(
            config.numExperts.isMultiple(of: config.nGroup),
            "AfMoE experts must divide evenly into groups"
        )
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "AfMoE top-k cannot exceed expert count"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= config.nGroup,
            "AfMoE kept-group count must be within group count"
        )

        self.expertCount = config.numExperts
        self.selectedExpertCount = config.numExpertsPerToken
        self.groupCount = config.nGroup
        self.keptGroupCount = keptGroupCount
        self.normalizesSelectedProbabilities = config.routeNorm
        self.routedScalingFactor = config.routeScale
        self.scoreFunction = AfMoEScoreFunction(config.scoreFunc)
    }

    internal var expertsPerGroup: Int {
        expertCount / groupCount
    }

    private var droppedGroupCount: Int {
        groupCount - keptGroupCount
    }

    internal func route(
        logits: MLXArray,
        expertBias: MLXArray,
        outputDType: DType
    ) -> (scores: MLXArray, indices: MLXArray) {
        let originalScores = scoreFunction.scores(from: logits)
        var selectionScores = originalScores + expertBias

        if groupCount > 1, droppedGroupCount > 0 {
            selectionScores = unflatten(selectionScores, axis: -1, shape: [groupCount, -1])
            let groupScores = top(selectionScores, k: min(2, expertsPerGroup), axis: -1)
                .sum(axis: -1, keepDims: true)
            let droppedGroups = argPartition(
                groupScores,
                kth: droppedGroupCount - 1,
                axis: -2
            )[.ellipsis, ..<droppedGroupCount, 0...]
            selectionScores = putAlong(
                selectionScores,
                stopGradient(droppedGroups),
                values: MLXArray(0.0),
                axis: -2
            )
            selectionScores = flattened(selectionScores, start: -2, end: -1)
        }

        let indices = argPartition(
            -selectionScores,
            kth: selectedExpertCount - 1,
            axis: -1
        )[.ellipsis, ..<selectedExpertCount]
        var selectedScores = takeAlong(originalScores, indices, axis: -1)

        if selectedExpertCount > 1, normalizesSelectedProbabilities {
            selectedScores = selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
        }

        return ((selectedScores * routedScalingFactor).asType(outputDType), indices)
    }
}

private enum AfMoEExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal struct AfMoEExpertPackingPlan: Equatable, Sendable {
    internal let layerPlan: AfMoELayerPlan
    internal let expertCount: Int

    internal init(_ config: AfMoEConfiguration) {
        self.layerPlan = AfMoELayerPlan(config)
        self.expertCount = config.numExperts
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights

        for layerIndex in 0 ..< layerPlan.layerCount where layerPlan.usesSparseExperts(
            layerIndex: layerIndex
        ) {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in AfMoEExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection.rawValue).\(tensorName)"
                    guard packed[firstKey] != nil else { continue }

                    let tensors = (0 ..< expertCount).map { expertIndex in
                        packed.removeValue(
                            forKey: "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                        )!
                    }
                    packed["\(prefix).experts.\(projection.rawValue).\(tensorName)"] =
                        stacked(tensors)
                }
            }
        }

        return packed
    }
}

// MARK: - Layers

internal final class AfMoEAttention: Module {
    let layout: AfMoEAttentionLayout

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear

    let rope: RoPELayer?

    init(_ config: AfMoEConfiguration, kind: AfMoEAttentionKind) {
        self.layout = AfMoEAttentionLayout(config, kind: kind)

        _query.wrappedValue = Linear(config.hiddenSize, layout.queryProjectionSize, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, layout.keyValueProjectionSize, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, layout.keyValueProjectionSize, bias: false)
        _output.wrappedValue = Linear(layout.queryProjectionSize, config.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: config.rmsNormEps)
        _gate.wrappedValue = Linear(config.hiddenSize, layout.queryProjectionSize, bias: false)

        if layout.usesRotaryPosition {
            self.rope = initializeRope(
                dims: layout.headDimensions,
                base: config.ropeTheta,
                traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: nil
            )
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let tokenCount = x.dim(1)

        var queryStates = query(x)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keyStates = key(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let valueStates = value(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        queryStates = queryNorm(queryStates)
        keyStates = keyNorm(keyStates)

        if let rope {
            queryStates = applyRotaryPosition(rope, to: queryStates, cache: cache)
            keyStates = applyRotaryPosition(rope, to: keyStates, cache: cache)
        }

        var attentionOutput = attentionWithCacheUpdate(
            queries: queryStates,
            keys: keyStates,
            values: valueStates,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        attentionOutput = attentionOutput * sigmoid(gate(x))
        return output(attentionOutput)
    }
}

internal final class AfMoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

internal final class AfMoERouter: Module {
    @ModuleInfo(key: "gate") var gate: Linear

    init(_ config: AfMoEConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gate(x)
    }
}

internal final class AfMoESparseBlock: Module, UnaryLayer {
    let routingPlan: AfMoERoutingPlan

    @ModuleInfo var router: AfMoERouter
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: AfMoEMLP?

    init(_ config: AfMoEConfiguration) {
        self.routingPlan = AfMoERoutingPlan(config)

        _router.wrappedValue = AfMoERouter(config)
        _expertBias.wrappedValue = MLXArray.zeros([config.numExperts])
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )

        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = AfMoEMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.moeIntermediateSize * config.numSharedExperts
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = routingPlan.route(
            logits: router(x),
            expertBias: expertBias,
            outputDType: x.dtype
        )
        var output = experts(x, routed.indices)
        output = (output * routed.scores[.ellipsis, .newAxis]).sum(axis: -2).asType(output.dtype)
        if let sharedExperts {
            output = output + sharedExperts(x)
        }
        return output
    }
}

internal final class AfMoEDecoderLayer: Module {
    let attentionKind: AfMoEAttentionKind

    @ModuleInfo(key: "self_attn") var attention: AfMoEAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_mlp_layernorm") var preMLPLayerNorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMLPLayerNorm: RMSNorm

    init(_ config: AfMoEConfiguration, layerIndex: Int, layerPlan: AfMoELayerPlan) {
        self.attentionKind = layerPlan.attentionKind(layerIndex: layerIndex)

        _attention.wrappedValue = AfMoEAttention(config, kind: attentionKind)
        if layerPlan.usesSparseExperts(layerIndex: layerIndex) {
            _feedForward.wrappedValue = AfMoESparseBlock(config)
        } else {
            _feedForward.wrappedValue = AfMoEMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _preMLPLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postMLPLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var residual = attention(inputLayerNorm(x), mask: mask, cache: cache)
        residual = postAttentionLayerNorm(residual)
        let hidden = x + residual

        residual = feedForward(preMLPLayerNorm(hidden))
        residual = postMLPLayerNorm(residual)
        return hidden + residual
    }
}

internal final class AfMoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [AfMoEDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerPlan: AfMoELayerPlan
    let usesMupScaling: Bool
    let embeddingScale: Float

    init(_ config: AfMoEConfiguration) {
        precondition(config.vocabularySize > 0, "AfMoE vocabulary size must be positive")

        let plan = AfMoELayerPlan(config)
        self.layerPlan = plan
        self.usesMupScaling = config.mupEnabled
        self.embeddingScale = sqrt(Float(config.hiddenSize))

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< plan.layerCount).map { layerIndex in
            AfMoEDecoderLayer(config, layerIndex: layerIndex, layerPlan: plan)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs)
        if usesMupScaling {
            hidden = hidden * embeddingScale
        }

        let fullMask = createAttentionMask(
            h: hidden,
            cache: cache?[layerPlan.firstFullLayer]
        )
        let slidingMask = layerPlan.firstSlidingLayer.map { layerIndex in
            createAttentionMask(
                h: hidden,
                cache: cache?[layerIndex],
                windowSize: layerPlan.slidingWindow
            )
        }

        for (layerIndex, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                layer.attentionKind == .sliding ? (slidingMask ?? fullMask) : fullMask
            hidden = layer(hidden, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hidden)
    }
}

internal final class AfMoEModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") fileprivate var model: AfMoEBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    private let configuration: AfMoEConfiguration
    let modelType: String

    internal init(_ config: AfMoEConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.modelType = config.modelType
        _model.wrappedValue = AfMoEBackbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
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
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        sanitized = sanitized.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }
        return AfMoEExpertPackingPlan(configuration).pack(sanitized)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        AfMoELayerPlan(configuration).attentionKinds.map { kind in
            switch kind {
            case .full:
                KVCacheSimple()
            case .sliding:
                RotatingKVCache(maxSize: configuration.slidingWindow)
            }
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}

// MARK: - LoRA

extension AfMoEModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
