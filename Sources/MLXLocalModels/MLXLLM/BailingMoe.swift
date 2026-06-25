import Foundation
import MLX
import MLXNN

// MARK: - Plans

internal struct BailingMoeAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let rotaryDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: BailingMoeConfiguration) {
        let headDimensions = config.hiddenSize / config.attentionHeads

        precondition(config.attentionHeads > 0, "Bailing MoE attention heads must be positive")
        precondition(config.kvHeads > 0, "Bailing MoE KV heads must be positive")
        precondition(headDimensions > 0, "Bailing MoE head dimensions must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "Bailing MoE attention heads must group KV heads"
        )
        precondition(
            config.partialRotaryFactor > 0,
            "Bailing MoE partial rotary factor must be positive"
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
    internal var queryKeyValueProjectionSize: Int {
        queryProjectionSize + 2 * keyValueProjectionSize
    }
}

internal struct BailingMoeLayerPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int
    internal let firstSparseLayer: Int

    internal init(_ config: BailingMoeConfiguration) {
        precondition(config.hiddenLayers > 0, "Bailing MoE must have at least one layer")
        precondition(config.firstKDenseReplace >= 0, "Bailing MoE dense-layer count is negative")
        precondition(
            config.firstKDenseReplace <= config.hiddenLayers,
            "Bailing MoE dense-layer count exceeds layer count"
        )

        self.layerCount = config.hiddenLayers
        self.expertCount = config.numExperts
        self.firstSparseLayer = config.firstKDenseReplace
    }

    internal func usesSparseExperts(layerIndex: Int) -> Bool {
        expertCount > 0 && layerIndex >= firstSparseLayer
    }
}

internal enum BailingMoeScoreFunction: String, Sendable {
    case sigmoid
    case softmax

    internal init(_ value: String) {
        self = BailingMoeScoreFunction(rawValue: value) ?? .softmax
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

internal struct BailingMoeRoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizesSelectedProbabilities: Bool
    internal let routedScalingFactor: Float
    internal let scoreFunction: BailingMoeScoreFunction
    internal let usesExpertBias: Bool

    internal init(_ config: BailingMoeConfiguration) {
        let keptGroupCount = min(config.topkGroup, config.nGroup)

        precondition(config.numExperts > 0, "Bailing MoE expert count must be positive")
        precondition(
            config.numExpertsPerToken > 0,
            "Bailing MoE selected expert count must be positive"
        )
        precondition(config.nGroup > 0, "Bailing MoE group count must be positive")
        precondition(
            config.numExperts.isMultiple(of: config.nGroup),
            "Bailing MoE experts must divide evenly into groups"
        )
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "Bailing MoE cannot select more experts than it owns"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= config.nGroup,
            "Bailing MoE kept-group count must be within group count"
        )

        self.expertCount = config.numExperts
        self.selectedExpertCount = config.numExpertsPerToken
        self.groupCount = config.nGroup
        self.keptGroupCount = keptGroupCount
        self.normalizesSelectedProbabilities = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.scoreFunction = BailingMoeScoreFunction(config.scoreFunction)
        self.usesExpertBias = config.moeRouterEnableExpertBias
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
        var selectionScores = originalScores
        if usesExpertBias {
            selectionScores = selectionScores + expertBias
        }

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

private enum BailingMoeExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal struct BailingMoeExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: BailingMoeConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.numExperts
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in BailingMoeExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection.rawValue).\(tensorName)"
                    guard packed[firstKey] != nil else { continue }

                    let tensors = (0 ..< expertCount).map { expertIndex in
                        packed.removeValue(
                            forKey: "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                        )!
                    }
                    packed["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        MLX.stacked(tensors)
                }
            }
        }

        return packed
    }
}

// MARK: - Model Components

internal final class BailingMoeAttention: Module {
    let layout: BailingMoeAttentionLayout

    @ModuleInfo(key: "query_key_value") var queryKeyValue: Linear
    @ModuleInfo(key: "dense") var output: Linear
    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?

    let rope: RoPELayer

    init(_ config: BailingMoeConfiguration) {
        self.layout = BailingMoeAttentionLayout(config)

        _queryKeyValue.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryKeyValueProjectionSize,
            bias: config.useQKVBias
        )
        _output.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: config.useBias
        )

        if config.useQKNorm {
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

        let projected = queryKeyValue(x)
        let splitPoints = [
            layout.queryProjectionSize,
            layout.queryProjectionSize + layout.keyValueProjectionSize
        ]
        let splits = split(projected, indices: splitPoints, axis: -1)

        var queryStates = splits[0]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
        var keyStates = splits[1]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
        let valueStates = splits[2]
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

internal final class BailingMoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: BailingMoeConfiguration, hiddenDimensions: Int? = nil) {
        let hiddenDimensions = hiddenDimensions ?? config.intermediateSize
        _gate.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: config.useBias)
        _down.wrappedValue = Linear(hiddenDimensions, config.hiddenSize, bias: config.useBias)
        _up.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: config.useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

internal final class BailingMoeRouter: Module {
    let routingPlan: BailingMoeRoutingPlan

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ config: BailingMoeConfiguration) {
        self.routingPlan = BailingMoeRoutingPlan(config)

        _gate.wrappedValue = Linear(config.hiddenSize, routingPlan.expertCount, bias: false)
        _expertBias.wrappedValue = MLXArray.zeros([routingPlan.expertCount])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gate(x)
    }

    func route(_ x: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        routingPlan.route(
            logits: gate(x),
            expertBias: expertBias,
            outputDType: x.dtype
        )
    }
}

internal final class BailingMoeSparseBlock: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var router: BailingMoeRouter
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingMoeMLP?

    init(_ config: BailingMoeConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            bias: config.useBias
        )
        _router.wrappedValue = BailingMoeRouter(config)

        if config.numSharedExperts > 0 {
            let sharedDimensions = (
                config.moeSharedExpertIntermediateSize ?? config.moeIntermediateSize
            ) * config.numSharedExperts
            _sharedExperts.wrappedValue = BailingMoeMLP(
                config,
                hiddenDimensions: sharedDimensions
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = router.route(x)
        let expertOutput = switchMLP(x, routed.indices)
        var output = (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            output = output + sharedExperts(x)
        }
        return output
    }
}

internal final class BailingMoeBlock: Module {
    @ModuleInfo(key: "attention") fileprivate var attention: BailingMoeAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: BailingMoeConfiguration, layerIndex: Int, layerPlan: BailingMoeLayerPlan) {
        _attention.wrappedValue = BailingMoeAttention(config)
        if layerPlan.usesSparseExperts(layerIndex: layerIndex) {
            _feedForward.wrappedValue = BailingMoeSparseBlock(config)
        } else {
            _feedForward.wrappedValue = BailingMoeMLP(config)
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

internal final class BailingMoeBackbone: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo var layers: [BailingMoeBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: BailingMoeConfiguration) {
        precondition(config.vocabularySize > 0, "Bailing MoE vocabulary size must be positive")

        let layerPlan = BailingMoeLayerPlan(config)
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            BailingMoeBlock(config, layerIndex: layerIndex, layerPlan: layerPlan)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = wordEmbeddings(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (layerIndex, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hidden)
    }
}

internal final class BailingMoeModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: BailingMoeBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    private let configuration: BailingMoeConfiguration
    let modelType: String

    internal init(_ config: BailingMoeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.modelType = config.modelType
        _model.wrappedValue = BailingMoeBackbone(config)

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
        sanitized = sanitized.filter { !$0.key.contains("attention.rotary_emb.inv_freq") }
        return BailingMoeExpertPackingPlan(configuration).pack(sanitized)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.wordEmbeddings.asLinear(hiddenStates)
    }
}

// MARK: - Configuration

internal struct BailingMoeConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int?
    var moeIntermediateSize: Int
    var numExperts: Int
    var numSharedExperts: Int
    var normTopkProb: Bool
    var attentionHeads: Int
    var numExpertsPerToken: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var vocabularySize: Int
    var firstKDenseReplace: Int
    var ropeScaling: [String: StringOrNumber]?
    var useBias: Bool
    var useQKVBias: Bool
    var useQKNorm: Bool
    var tieWordEmbeddings: Bool
    var partialRotaryFactor: Float
    var moeRouterEnableExpertBias: Bool
    var routedScalingFactor: Float
    var scoreFunction: String
    var nGroup: Int
    var topkGroup: Int
    var moeSharedExpertIntermediateSize: Int?

    internal init(
        modelType: String = "bailing_moe",
        hiddenSize: Int,
        intermediateSize: Int,
        maxPositionEmbeddings: Int? = nil,
        moeIntermediateSize: Int? = nil,
        numExperts: Int,
        numSharedExperts: Int = 0,
        normTopkProb: Bool = false,
        attentionHeads: Int,
        numExpertsPerToken: Int,
        hiddenLayers: Int,
        kvHeads: Int? = nil,
        rmsNormEps: Float,
        ropeTheta: Float = 10_000,
        vocabularySize: Int,
        firstKDenseReplace: Int = 0,
        ropeScaling: [String: StringOrNumber]? = nil,
        useBias: Bool = false,
        useQKVBias: Bool = false,
        useQKNorm: Bool = false,
        tieWordEmbeddings: Bool = false,
        partialRotaryFactor: Float = 1.0,
        moeRouterEnableExpertBias: Bool = false,
        routedScalingFactor: Float = 1.0,
        scoreFunction: String = "softmax",
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        moeSharedExpertIntermediateSize: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.moeIntermediateSize = moeIntermediateSize ?? intermediateSize
        self.numExperts = numExperts
        self.numSharedExperts = numSharedExperts
        self.normTopkProb = normTopkProb
        self.attentionHeads = attentionHeads
        self.numExpertsPerToken = numExpertsPerToken
        self.hiddenLayers = hiddenLayers
        self.kvHeads = kvHeads ?? attentionHeads
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.vocabularySize = vocabularySize
        self.firstKDenseReplace = firstKDenseReplace
        self.ropeScaling = ropeScaling
        self.useBias = useBias
        self.useQKVBias = useQKVBias
        self.useQKNorm = useQKNorm
        self.tieWordEmbeddings = tieWordEmbeddings
        self.partialRotaryFactor = partialRotaryFactor
        self.moeRouterEnableExpertBias = moeRouterEnableExpertBias
        self.routedScalingFactor = routedScalingFactor
        self.scoreFunction = scoreFunction
        self.nGroup = nGroup
        self.topkGroup = topkGroup ?? nGroup
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case numSharedExperts = "num_shared_experts"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case numExpertsPerToken = "num_experts_per_tok"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case ropeScaling = "rope_scaling"
        case useBias = "use_bias"
        case useQKVBias = "use_qkv_bias"
        case useQKNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunction = "score_function"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        let nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "bailing_moe",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            intermediateSize: intermediateSize,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ) ?? intermediateSize,
            numExperts: try container.decode(Int.self, forKey: .numExperts),
            numSharedExperts: try container.decodeIfPresent(Int.self, forKey: .numSharedExperts)
                ?? 0,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false,
            attentionHeads: attentionHeads,
            numExpertsPerToken: try container.decode(Int.self, forKey: .numExpertsPerToken),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads,
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            firstKDenseReplace: try container.decodeIfPresent(
                Int.self,
                forKey: .firstKDenseReplace
            ) ?? 0,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            useBias: try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            useQKVBias: try container.decodeIfPresent(Bool.self, forKey: .useQKVBias)
                ?? false,
            useQKNorm: try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 1.0,
            moeRouterEnableExpertBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .moeRouterEnableExpertBias
            ) ?? false,
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 1.0,
            scoreFunction: try container.decodeIfPresent(String.self, forKey: .scoreFunction)
                ?? "softmax",
            nGroup: nGroup,
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? nGroup,
            moeSharedExpertIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeSharedExpertIntermediateSize
            )
        )
    }
}

// MARK: - LoRA

extension BailingMoeModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["query_key_value"]) }
    }
}
