import Foundation
import MLX
import MLXNN

// MARK: - Plans

internal struct GLM4MoEAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let rotaryDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: GLM4MoEConfiguration) {
        let headDimensions = config.headDim > 0
            ? config.headDim
            : config.hiddenSize / config.attentionHeads

        precondition(config.hiddenSize > 0, "GLM4 MoE hidden size must be positive")
        precondition(config.attentionHeads > 0, "GLM4 MoE attention heads must be positive")
        precondition(config.kvHeads > 0, "GLM4 MoE KV heads must be positive")
        precondition(headDimensions > 0, "GLM4 MoE head dimensions must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "GLM4 MoE attention heads must group KV heads"
        )
        precondition(
            config.partialRotaryFactor > 0,
            "GLM4 MoE partial rotary factor must be positive"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = headDimensions
        self.rotaryDimensions = max(1, Int(Float(headDimensions) * config.partialRotaryFactor))
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
    internal var outputProjectionInputSize: Int { queryProjectionSize }
}

internal struct GLM4MoELayerPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let firstSparseLayer: Int
    internal let routedExpertCount: Int?

    internal init(_ config: GLM4MoEConfiguration) {
        precondition(config.hiddenLayers > 0, "GLM4 MoE must have at least one layer")
        precondition(config.firstKDenseReplace >= 0, "GLM4 MoE dense-layer count is negative")
        precondition(
            config.firstKDenseReplace <= config.hiddenLayers,
            "GLM4 MoE dense-layer count exceeds layer count"
        )

        self.layerCount = config.hiddenLayers
        self.firstSparseLayer = config.firstKDenseReplace
        self.routedExpertCount = config.nRoutedExperts
    }

    internal func usesSparseExperts(layerIndex: Int) -> Bool {
        (routedExpertCount ?? 0) > 0 && layerIndex >= firstSparseLayer
    }
}

internal enum GLM4MoEScoreFunction: String, Equatable, Sendable {
    case sigmoid
    case softmax

    internal init(_ rawValue: String) {
        self = GLM4MoEScoreFunction(rawValue: rawValue) ?? .sigmoid
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

internal struct GLM4MoERoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizesSelectedProbabilities: Bool
    internal let routedScalingFactor: Float
    internal let scoreFunction: GLM4MoEScoreFunction

    internal init(_ config: GLM4MoEConfiguration) {
        let expertCount = config.nRoutedExperts ?? 0
        let keptGroupCount = min(config.topkGroup, config.nGroup)

        precondition(expertCount > 0, "GLM4 MoE routed expert count must be positive")
        precondition(
            config.topkMethod == "noaux_tc",
            "GLM4 MoE only supports noaux_tc routing"
        )
        precondition(config.numExpertsPerTok > 0, "GLM4 MoE top-k must be positive")
        precondition(config.nGroup > 0, "GLM4 MoE group count must be positive")
        precondition(
            expertCount.isMultiple(of: config.nGroup),
            "GLM4 MoE experts must divide evenly into groups"
        )
        precondition(
            config.numExpertsPerTok <= expertCount,
            "GLM4 MoE top-k cannot exceed routed expert count"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= config.nGroup,
            "GLM4 MoE kept-group count must be within group count"
        )

        self.expertCount = expertCount
        self.selectedExpertCount = config.numExpertsPerTok
        self.groupCount = config.nGroup
        self.keptGroupCount = keptGroupCount
        self.normalizesSelectedProbabilities = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.scoreFunction = GLM4MoEScoreFunction(config.scoringFunc)
    }

    internal var expertsPerGroup: Int {
        expertCount / groupCount
    }

    private var droppedGroupCount: Int {
        groupCount - keptGroupCount
    }

    internal func route(
        logits: MLXArray,
        correctionBias: MLXArray,
        outputDType: DType
    ) -> (scores: MLXArray, indices: MLXArray) {
        let originalScores = scoreFunction.scores(from: logits)
        var selectionScores = originalScores + correctionBias

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

private enum GLM4MoEExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal struct GLM4MoEExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: GLM4MoEConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.nRoutedExperts ?? 0
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        guard expertCount > 0 else { return weights }

        var packed = weights
        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in GLM4MoEExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection.rawValue).\(tensorName)"
                    guard packed[firstKey] != nil else { continue }

                    let tensors = (0 ..< expertCount).map { expertIndex in
                        packed.removeValue(
                            forKey: "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                        )!
                    }
                    packed["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        stacked(tensors)
                }
            }
        }
        return packed
    }
}

// MARK: - Model Components

internal final class GLM4MoEAttention: Module {
    let layout: GLM4MoEAttentionLayout

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm?

    let rope: RoPELayer

    init(_ config: GLM4MoEConfiguration) {
        self.layout = GLM4MoEAttentionLayout(config)

        _query.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        _key.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _value.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _output.wrappedValue = Linear(
            layout.outputProjectionInputSize,
            config.hiddenSize,
            bias: false
        )

        if config.useQkNorm {
            _queryNorm.wrappedValue = RMSNorm(
                dimensions: layout.headDimensions,
                eps: config.rmsNormEps
            )
            _keyNorm.wrappedValue = RMSNorm(
                dimensions: layout.headDimensions,
                eps: config.rmsNormEps
            )
        }

        self.rope = initializeRope(
            dims: layout.rotaryDimensions,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
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
        var keyStates = key(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
        let valueStates = value(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        if let queryNorm {
            queryStates = queryNorm(queryStates)
        }
        if let keyNorm {
            keyStates = keyNorm(keyStates)
        }

        queryStates = queryStates.transposed(0, 2, 1, 3)
        keyStates = keyStates.transposed(0, 2, 1, 3)

        queryStates = applyRotaryPosition(rope, to: queryStates, cache: cache)
        keyStates = applyRotaryPosition(rope, to: keyStates, cache: cache)

        let attentionOutput = attentionWithCacheUpdate(
            queries: queryStates,
            keys: keyStates,
            values: valueStates,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return output(attentionOutput)
    }
}

internal final class GLM4MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: GLM4MoEConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let hiddenSize = hiddenSize ?? config.hiddenSize
        let intermediateSize = intermediateSize ?? config.intermediateSize

        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

internal final class GLM4MoEGate: Module {
    let routingPlan: GLM4MoERoutingPlan

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ config: GLM4MoEConfiguration) {
        self.routingPlan = GLM4MoERoutingPlan(config)
        _weight.wrappedValue = MLXArray.zeros([routingPlan.expertCount, config.hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([routingPlan.expertCount])
        super.init()
    }

    func route(_ x: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        routingPlan.route(
            logits: x.matmul(weight.T),
            correctionBias: correctionBias,
            outputDType: x.dtype
        )
    }
}

internal final class GLM4MoESparseBlock: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: GLM4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoEMLP?

    init(_ config: GLM4MoEConfiguration) {
        let expertCount = config.nRoutedExperts ?? 0
        precondition(expertCount > 0, "GLM4 MoE sparse block requires routed experts")

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: expertCount
        )
        _gate.wrappedValue = GLM4MoEGate(config)

        if let sharedExpertCount = config.nSharedExperts, sharedExpertCount > 0 {
            _sharedExperts.wrappedValue = GLM4MoEMLP(
                config,
                intermediateSize: config.moeIntermediateSize * sharedExpertCount
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = gate.route(x)
        let expertOutput = switchMLP(x, routed.indices)
        var output = (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            output = output + sharedExperts(x)
        }
        return output
    }
}

internal final class GLM4MoEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoEAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM4MoEConfiguration, layerIndex: Int, layerPlan: GLM4MoELayerPlan) {
        _attention.wrappedValue = GLM4MoEAttention(config)
        if layerPlan.usesSparseExperts(layerIndex: layerIndex) {
            _feedForward.wrappedValue = GLM4MoESparseBlock(config)
        } else {
            _feedForward.wrappedValue = GLM4MoEMLP(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hidden = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        hidden = hidden + feedForward(postAttentionLayerNorm(hidden))
        return hidden
    }
}

internal final class GLM4MoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [GLM4MoEDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: GLM4MoEConfiguration) {
        precondition(config.vocabularySize > 0, "GLM4 MoE vocabulary size must be positive")

        let layerPlan = GLM4MoELayerPlan(config)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            GLM4MoEDecoderLayer(config, layerIndex: layerIndex, layerPlan: layerPlan)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (layerIndex, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hidden)
    }
}

internal final class GLM4MoEModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") internal var model: GLM4MoEBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    private let configuration: GLM4MoEConfiguration
    let modelType: String

    internal init(_ config: GLM4MoEConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.modelType = config.modelType
        _model.wrappedValue = GLM4MoEBackbone(config)

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

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { _ in KVCacheSimple() }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        sanitized = sanitized.filter {
            !$0.key.contains("attention.rotary_emb.inv_freq")
                && !$0.key.hasPrefix("model.layers.\(configuration.hiddenLayers)")
        }
        return GLM4MoEExpertPackingPlan(configuration).pack(sanitized)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}

// MARK: - Configuration

internal struct GLM4MoEConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var moeIntermediateSize: Int
    var normTopkProb: Bool
    var attentionHeads: Int
    var nGroup: Int
    var headDim: Int
    var topkGroup: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var numExpertsPerTok: Int
    var firstKDenseReplace: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var useQkNorm: Bool
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var partialRotaryFactor: Float
    var scoringFunc: String
    var topkMethod: String

    internal init(
        modelType: String = "glm4_moe",
        vocabularySize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        maxPositionEmbeddings: Int = 32_768,
        moeIntermediateSize: Int? = nil,
        normTopkProb: Bool = false,
        attentionHeads: Int,
        nGroup: Int = 1,
        headDim: Int? = nil,
        topkGroup: Int? = nil,
        nSharedExperts: Int? = nil,
        nRoutedExperts: Int? = nil,
        routedScalingFactor: Float = 1,
        numExpertsPerTok: Int = 1,
        firstKDenseReplace: Int = 0,
        hiddenLayers: Int,
        kvHeads: Int? = nil,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        useQkNorm: Bool = false,
        tieWordEmbeddings: Bool = false,
        attentionBias: Bool = false,
        partialRotaryFactor: Float = 1.0,
        scoringFunc: String = "sigmoid",
        topkMethod: String = "noaux_tc"
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.moeIntermediateSize = moeIntermediateSize ?? intermediateSize
        self.normTopkProb = normTopkProb
        self.attentionHeads = attentionHeads
        self.nGroup = nGroup
        self.headDim = headDim ?? (hiddenSize / attentionHeads)
        self.topkGroup = topkGroup ?? nGroup
        self.nSharedExperts = nSharedExperts
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = routedScalingFactor
        self.numExpertsPerTok = numExpertsPerTok
        self.firstKDenseReplace = firstKDenseReplace
        self.hiddenLayers = hiddenLayers
        self.kvHeads = kvHeads ?? attentionHeads
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.useQkNorm = useQkNorm
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.partialRotaryFactor = partialRotaryFactor
        self.scoringFunc = scoringFunc
        self.topkMethod = topkMethod
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case nGroup = "n_group"
        case headDim = "head_dim"
        case topkGroup = "topk_group"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case useQkNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        let nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "glm4_moe",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 32_768,
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ) ?? intermediateSize,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false,
            attentionHeads: attentionHeads,
            nGroup: nGroup,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim)
                ?? (hiddenSize / attentionHeads),
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? nGroup,
            nSharedExperts: try container.decodeIfPresent(Int.self, forKey: .nSharedExperts),
            nRoutedExperts: try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts),
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 1,
            numExpertsPerTok: try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
                ?? 1,
            firstKDenseReplace: try container.decodeIfPresent(
                Int.self,
                forKey: .firstKDenseReplace
            ) ?? 0,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            useQkNorm: try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 1.0,
            scoringFunc: try container.decodeIfPresent(String.self, forKey: .scoringFunc)
                ?? "sigmoid",
            topkMethod: try container.decodeIfPresent(String.self, forKey: .topkMethod)
                ?? "noaux_tc"
        )
    }
}

// MARK: - LoRA

extension GLM4MoEModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
