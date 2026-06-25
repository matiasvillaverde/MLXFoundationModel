import Foundation
import MLX
import MLXNN

// MARK: - Planning

internal enum NemotronHBlockKind: String, Sendable, Equatable {
    case mamba = "M"
    case attention = "*"
    case feedForward = "-"
    case routedFeedForward = "E"

    internal var needsCache: Bool {
        switch self {
        case .mamba, .attention:
            return true
        case .feedForward, .routedFeedForward:
            return false
        }
    }

    internal static func parse(_ symbol: Character) -> NemotronHBlockKind? {
        Self(rawValue: String(symbol))
    }
}

internal enum NemotronHCacheKind: Sendable, Equatable {
    case mamba
    case attention(kvHeads: Int)
}

internal struct NemotronHLayerPlan: Sendable, Equatable {
    internal let kinds: [NemotronHBlockKind]
    internal let cacheKinds: [NemotronHCacheKind]
    internal let cacheIndexByLayer: [Int?]
    internal let firstAttentionCacheIndex: Int?
    internal let firstMambaCacheIndex: Int?

    internal init(_ configuration: NemotronHConfiguration) {
        precondition(
            configuration.blockPattern.count == configuration.numHiddenLayers,
            "hybrid_override_pattern must match num_hidden_layers"
        )

        var cacheKinds = [NemotronHCacheKind]()
        var cacheIndexByLayer = [Int?]()
        var firstAttentionCacheIndex: Int?
        var firstMambaCacheIndex: Int?

        for kind in configuration.blockPattern {
            switch kind {
            case .mamba:
                if firstMambaCacheIndex == nil {
                    firstMambaCacheIndex = cacheKinds.count
                }
                cacheIndexByLayer.append(cacheKinds.count)
                cacheKinds.append(.mamba)
            case .attention:
                if firstAttentionCacheIndex == nil {
                    firstAttentionCacheIndex = cacheKinds.count
                }
                cacheIndexByLayer.append(cacheKinds.count)
                cacheKinds.append(.attention(kvHeads: configuration.numKeyValueHeads))
            case .feedForward, .routedFeedForward:
                cacheIndexByLayer.append(nil)
            }
        }

        self.kinds = configuration.blockPattern
        self.cacheKinds = cacheKinds
        self.cacheIndexByLayer = cacheIndexByLayer
        self.firstAttentionCacheIndex = firstAttentionCacheIndex
        self.firstMambaCacheIndex = firstMambaCacheIndex
    }

    internal var kvHeads: [Int] {
        cacheKinds.map { cacheKind in
            switch cacheKind {
            case .mamba:
                return 0
            case .attention(let kvHeads):
                return kvHeads
            }
        }
    }
}

internal struct NemotronHAttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let queryDimensions: Int
    internal let keyValueDimensions: Int
    internal let scale: Float

    internal init(_ configuration: NemotronHConfiguration) {
        let headDimensions = configuration.headDim
            ?? configuration.hiddenSize / configuration.numAttentionHeads

        precondition(configuration.hiddenSize > 0, "hidden_size must be positive")
        precondition(configuration.numAttentionHeads > 0, "num_attention_heads must be positive")
        precondition(configuration.numKeyValueHeads > 0, "num_key_value_heads must be positive")
        precondition(headDimensions > 0, "head_dim must be positive")
        precondition(
            configuration.hiddenSize == configuration.numAttentionHeads * headDimensions,
            "hidden_size must equal num_attention_heads * head_dim"
        )
        precondition(
            configuration.numAttentionHeads.isMultiple(of: configuration.numKeyValueHeads),
            "attention heads must group key-value heads"
        )

        self.hiddenSize = configuration.hiddenSize
        self.attentionHeads = configuration.numAttentionHeads
        self.keyValueHeads = configuration.numKeyValueHeads
        self.headDimensions = headDimensions
        self.queryDimensions = configuration.numAttentionHeads * headDimensions
        self.keyValueDimensions = configuration.numKeyValueHeads * headDimensions
        self.scale = pow(Float(headDimensions), -0.5)
    }
}

internal struct NemotronHMambaLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let heads: Int
    internal let headDimensions: Int
    internal let intermediateSize: Int
    internal let groups: Int
    internal let stateDimensions: Int
    internal let convolutionKernelSize: Int
    internal let convolutionDimensions: Int
    internal let projectionDimensions: Int
    internal let gatedNormGroupSize: Int
    internal let timeStepLimit: (Float, Float)

    internal init(_ configuration: NemotronHConfiguration) {
        precondition(configuration.hiddenSize > 0, "hidden_size must be positive")
        precondition(configuration.mambaNumHeads > 0, "mamba_num_heads must be positive")
        precondition(configuration.mambaHeadDim > 0, "mamba_head_dim must be positive")
        precondition(configuration.ssmStateSize > 0, "ssm_state_size must be positive")
        precondition(configuration.convKernel > 0, "conv_kernel must be positive")
        precondition(configuration.nGroups > 0, "n_groups must be positive")
        precondition(
            configuration.mambaNumHeads.isMultiple(of: configuration.nGroups),
            "mamba_num_heads must be divisible by n_groups"
        )

        let intermediateSize = configuration.mambaNumHeads * configuration.mambaHeadDim
        precondition(
            intermediateSize.isMultiple(of: configuration.nGroups),
            "Mamba intermediate size must be divisible by n_groups"
        )

        let convolutionDimensions = intermediateSize
            + 2 * configuration.nGroups * configuration.ssmStateSize

        self.hiddenSize = configuration.hiddenSize
        self.heads = configuration.mambaNumHeads
        self.headDimensions = configuration.mambaHeadDim
        self.intermediateSize = intermediateSize
        self.groups = configuration.nGroups
        self.stateDimensions = configuration.ssmStateSize
        self.convolutionKernelSize = configuration.convKernel
        self.convolutionDimensions = convolutionDimensions
        self.projectionDimensions = intermediateSize + convolutionDimensions
            + configuration.mambaNumHeads
        self.gatedNormGroupSize = intermediateSize / configuration.nGroups
        self.timeStepLimit = (configuration.timeStepLimitMin, configuration.timeStepLimitMax)
    }

    internal static func == (lhs: NemotronHMambaLayout, rhs: NemotronHMambaLayout) -> Bool {
        lhs.hiddenSize == rhs.hiddenSize
            && lhs.heads == rhs.heads
            && lhs.headDimensions == rhs.headDimensions
            && lhs.intermediateSize == rhs.intermediateSize
            && lhs.groups == rhs.groups
            && lhs.stateDimensions == rhs.stateDimensions
            && lhs.convolutionKernelSize == rhs.convolutionKernelSize
            && lhs.convolutionDimensions == rhs.convolutionDimensions
            && lhs.projectionDimensions == rhs.projectionDimensions
            && lhs.gatedNormGroupSize == rhs.gatedNormGroupSize
            && lhs.timeStepLimit.0 == rhs.timeStepLimit.0
            && lhs.timeStepLimit.1 == rhs.timeStepLimit.1
    }
}

internal struct NemotronHMoEPlan: Sendable, Equatable {
    internal let routedExperts: Int
    internal let expertsPerToken: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizesTopKProbabilities: Bool
    internal let routedScalingFactor: Float

    internal init(_ configuration: NemotronHConfiguration) {
        let keptGroupCount = min(configuration.topkGroup, configuration.nGroup)

        precondition(configuration.nRoutedExperts > 0, "n_routed_experts must be positive")
        precondition(configuration.numExpertsPerTok > 0, "num_experts_per_tok must be positive")
        precondition(
            configuration.numExpertsPerTok <= configuration.nRoutedExperts,
            "num_experts_per_tok cannot exceed n_routed_experts"
        )
        precondition(configuration.nGroup > 0, "n_group must be positive")
        precondition(
            configuration.nRoutedExperts.isMultiple(of: configuration.nGroup),
            "n_routed_experts must divide evenly into n_group"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= configuration.nGroup,
            "topk_group must be within n_group"
        )

        self.routedExperts = configuration.nRoutedExperts
        self.expertsPerToken = configuration.numExpertsPerTok
        self.groupCount = configuration.nGroup
        self.keptGroupCount = keptGroupCount
        self.normalizesTopKProbabilities = configuration.normTopkProb
        self.routedScalingFactor = configuration.routedScalingFactor
    }

    internal var expertsPerGroup: Int {
        routedExperts / groupCount
    }

    internal var droppedGroupCount: Int {
        groupCount - keptGroupCount
    }

    internal func route(
        logits: MLXArray,
        correctionBias: MLXArray,
        outputDType: DType
    ) -> (indices: MLXArray, scores: MLXArray) {
        let originalScores = sigmoid(logits.asType(.float32))
        var selectionScores = originalScores + correctionBias

        if droppedGroupCount > 0 {
            let groupedScores = selectionScores.reshaped(
                selectionScores.dim(0),
                selectionScores.dim(1),
                groupCount,
                expertsPerGroup
            )
            let topGroupScores = top(groupedScores, k: min(2, expertsPerGroup), axis: -1)
                .sum(axis: -1, keepDims: true)
            let droppedGroups = argPartition(
                topGroupScores,
                kth: droppedGroupCount - 1,
                axis: -2
            )[.ellipsis, ..<droppedGroupCount, 0...]
            selectionScores = putAlong(
                groupedScores,
                stopGradient(droppedGroups),
                values: MLXArray(0.0),
                axis: -2
            )
            selectionScores = flattened(selectionScores, start: -2, end: -1)
        }

        let expertIndices = argPartition(
            -selectionScores,
            kth: expertsPerToken - 1,
            axis: -1
        )[.ellipsis, ..<expertsPerToken]
        var expertScores = takeAlong(originalScores, expertIndices, axis: -1)

        if expertsPerToken > 1, normalizesTopKProbabilities {
            expertScores = expertScores / (expertScores.sum(axis: -1, keepDims: true) + 1e-20)
        }

        return (expertIndices, (expertScores * routedScalingFactor).asType(outputDType))
    }
}

internal struct NemotronHSanitizerPlan {
    private let configuration: NemotronHConfiguration

    internal init(_ configuration: NemotronHConfiguration) {
        self.configuration = configuration
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        for (key, value) in sanitized where key.contains("conv1d.weight") {
            if value.ndim == 3, value.dim(-1) != 1 {
                sanitized[key] = value.swappedAxes(1, 2)
            }
        }

        for layerIndex in 0 ..< configuration.numHiddenLayers {
            let prefix = "backbone.layers.\(layerIndex).mixer"
            packExpertProjection(
                in: &sanitized,
                layerPrefix: prefix,
                sourceProjection: "up_proj",
                destinationProjection: "fc1"
            )
            packExpertProjection(
                in: &sanitized,
                layerPrefix: prefix,
                sourceProjection: "down_proj",
                destinationProjection: "fc2"
            )
        }

        return sanitized.filter { key, _ in
            !key.contains(".experts.")
        }
    }

    private func packExpertProjection(
        in weights: inout [String: MLXArray],
        layerPrefix: String,
        sourceProjection: String,
        destinationProjection: String
    ) {
        for suffix in ["weight", "scales", "biases"] {
            let expertWeights = (0 ..< configuration.nRoutedExperts).compactMap { expertIndex in
                weights.removeValue(
                    forKey: "\(layerPrefix).experts.\(expertIndex).\(sourceProjection).\(suffix)"
                )
            }
            guard expertWeights.count == configuration.nRoutedExperts else {
                continue
            }
            weights["\(layerPrefix).switch_mlp.\(destinationProjection).\(suffix)"] =
                stacked(expertWeights)
        }
    }
}

// MARK: - Shared Layers

private protocol NemotronHBlockOperator: Module {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

internal func nemotronHReluSquared(_ x: MLXArray) -> MLXArray {
    let positive = MLX.maximum(x, MLXArray(0))
    return positive * positive
}

internal final class NemotronHGatedRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    private let eps: Float
    private let groupSize: Int
    private let unitWeight: MLXArray

    init(dimensions: Int, eps: Float, groupSize: Int) {
        precondition(dimensions > 0, "dimensions must be positive")
        precondition(groupSize > 0, "groupSize must be positive")
        precondition(dimensions.isMultiple(of: groupSize), "dimensions must divide groupSize")

        self.eps = eps
        self.groupSize = groupSize
        self.unitWeight = MLXArray.ones([groupSize])
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray?) -> MLXArray {
        let gated = gate.map { hiddenStates * silu($0) } ?? hiddenStates
        let originalShape = gated.shape
        var groupedShape = Array(originalShape.dropLast())
        groupedShape.append(-1)
        groupedShape.append(groupSize)

        let normalized = MLXFast.rmsNorm(
            gated.reshaped(groupedShape),
            weight: unitWeight,
            eps: eps
        )
        return normalized.reshaped(originalShape) * weight
    }
}

internal final class NemotronHMambaMixer: Module, NemotronHBlockOperator {
    let layout: NemotronHMambaLayout

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear
    @ParameterInfo(key: "dt_bias") var timeStepBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var residualScale: MLXArray
    @ModuleInfo(key: "norm") var norm: NemotronHGatedRMSNorm

    init(_ configuration: NemotronHConfiguration) {
        self.layout = NemotronHMambaLayout(configuration)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: layout.convolutionDimensions,
            outputChannels: layout.convolutionDimensions,
            kernelSize: layout.convolutionKernelSize,
            groups: layout.convolutionDimensions,
            bias: configuration.useConvBias
        )
        _inProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.projectionDimensions,
            bias: configuration.mambaProjBias
        )
        _outProjection.wrappedValue = Linear(
            layout.intermediateSize,
            layout.hiddenSize,
            bias: configuration.mambaProjBias
        )
        _timeStepBias.wrappedValue = MLXArray.ones([layout.heads])
        _aLog.wrappedValue = log((MLXArray(0 ..< layout.heads).asType(.float32) + 1))
        _residualScale.wrappedValue = MLXArray.ones([layout.heads])
        _norm.wrappedValue = NemotronHGatedRMSNorm(
            dimensions: layout.intermediateSize,
            eps: configuration.layerNormEpsilon,
            groupSize: layout.gatedNormGroupSize
        )
        super.init()
    }

    private func applyConvolution(_ input: MLXArray, cache: MambaCache?) -> MLXArray {
        let batchSize = input.dim(0)
        let stateLength = max(0, layout.convolutionKernelSize - 1)
        let state = cache?[0] ?? MLXArray.zeros(
            [batchSize, stateLength, layout.convolutionDimensions],
            dtype: input.dtype
        )
        let padded = concatenated([state, input], axis: 1)

        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - stateLength)
            cache[0] = padded[0..., start ..< end, 0...]
        }

        return silu(conv1d(padded))
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let projected = inProjection(x)
        let parts = split(
            projected,
            indices: [
                layout.intermediateSize,
                layout.intermediateSize + layout.convolutionDimensions,
            ],
            axis: -1
        )
        let gate = parts[0]
        var convolutionInput = parts[1]
        let timeSteps = parts[2]

        if let ssmMask {
            convolutionInput = MLX.where(
                expandedDimensions(ssmMask, axis: -1),
                convolutionInput,
                MLXArray.zeros(like: convolutionInput)
            )
        }

        let convolutionOutput = applyConvolution(
            convolutionInput,
            cache: cache as? MambaCache
        )
        let convolutionParts = split(
            convolutionOutput,
            indices: [
                layout.intermediateSize,
                layout.intermediateSize + layout.groups * layout.stateDimensions,
            ],
            axis: -1
        )

        let hidden = convolutionParts[0].reshaped(
            x.dim(0),
            x.dim(1),
            layout.heads,
            layout.headDimensions
        )
        let inputState = convolutionParts[1].reshaped(
            x.dim(0),
            x.dim(1),
            layout.groups,
            layout.stateDimensions
        )
        let outputState = convolutionParts[2].reshaped(
            x.dim(0),
            x.dim(1),
            layout.groups,
            layout.stateDimensions
        )
        let dt = timeSteps.reshaped(x.dim(0), x.dim(1), layout.heads)
        let mambaCache = cache as? MambaCache

        let (output, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: inputState,
            C: outputState,
            D: residualScale,
            dt: dt,
            dtBias: timeStepBias,
            state: mambaCache?[1],
            timeStepLimit: layout.timeStepLimit,
            mask: ssmMask
        )

        if let mambaCache {
            mambaCache[1] = nextState
        }

        return outProjection(norm(output.flattened(start: 2), gate: gate))
    }
}

internal final class NemotronHAttention: Module, NemotronHBlockOperator {
    let layout: NemotronHAttentionLayout

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear

    init(_ configuration: NemotronHConfiguration) {
        self.layout = NemotronHAttentionLayout(configuration)

        _queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryDimensions,
            bias: configuration.attentionBias
        )
        _keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueDimensions,
            bias: configuration.attentionBias
        )
        _valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueDimensions,
            bias: configuration.attentionBias
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryDimensions,
            layout.hiddenSize,
            bias: configuration.attentionBias
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let tokenCount = x.dim(1)

        let queries = queryProjection(x)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let keys = keyProjection(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.scale,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(output)
    }
}

internal final class NemotronHDenseMLP: Module, UnaryLayer, NemotronHBlockOperator {
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ configuration: NemotronHConfiguration, intermediateSize: Int? = nil) {
        let intermediateSize = intermediateSize ?? configuration.intermediateSize
        _upProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            intermediateSize,
            bias: configuration.mlpBias
        )
        _downProjection.wrappedValue = Linear(
            intermediateSize,
            configuration.hiddenSize,
            bias: configuration.mlpBias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProjection(nemotronHReluSquared(upProjection(x)))
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

internal final class NemotronHExpertGate: Module {
    let plan: NemotronHMoEPlan

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var expertScoreCorrectionBias: MLXArray

    init(_ configuration: NemotronHConfiguration) {
        self.plan = NemotronHMoEPlan(configuration)
        _weight.wrappedValue = zeros([plan.routedExperts, configuration.hiddenSize])
        _expertScoreCorrectionBias.wrappedValue = zeros([plan.routedExperts])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        plan.route(
            logits: hiddenStates.matmul(weight.T),
            correctionBias: expertScoreCorrectionBias,
            outputDType: hiddenStates.dtype
        )
    }
}

internal final class NemotronHSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    init(inputDimensions: Int, hiddenDimensions: Int, expertCount: Int) {
        _fc1.wrappedValue = SwitchLinear(
            inputDims: inputDimensions,
            outputDims: hiddenDimensions,
            numExperts: expertCount,
            bias: false
        )
        _fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDimensions,
            outputDims: inputDimensions,
            numExperts: expertCount,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, expertIndices: MLXArray) -> MLXArray {
        let routedInput = expandedDimensions(x, axes: [-2, -3])
        let sortedDispatch = SwitchExpertDispatch.shouldSort(expertIndices: expertIndices)
        let permutation = sortedDispatch
            ? SwitchExpertPermutation(input: routedInput, expertIndices: expertIndices)
            : nil

        let input = permutation?.sortedInput ?? routedInput
        let indices = permutation?.sortedExpertIndices ?? expertIndices
        let output = fc2(
            nemotronHReluSquared(fc1(input, indices, sortedIndices: sortedDispatch)),
            indices,
            sortedIndices: sortedDispatch
        )

        return squeezed(permutation?.restore(output) ?? output, axis: -2)
    }
}

internal final class NemotronHMoE: Module, UnaryLayer, NemotronHBlockOperator {
    let plan: NemotronHMoEPlan

    @ModuleInfo(key: "gate") var gate: NemotronHExpertGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronHSwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHDenseMLP?

    init(_ configuration: NemotronHConfiguration) {
        self.plan = NemotronHMoEPlan(configuration)
        _gate.wrappedValue = NemotronHExpertGate(configuration)
        _switchMLP.wrappedValue = NemotronHSwitchMLP(
            inputDimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.moeIntermediateSize,
            expertCount: plan.routedExperts
        )

        if let sharedExpertCount = configuration.nSharedExperts, sharedExpertCount > 0 {
            _sharedExperts.wrappedValue = NemotronHDenseMLP(
                configuration,
                intermediateSize: configuration.moeSharedExpertIntermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let route = gate(x)
        var output = switchMLP(x, expertIndices: route.indices)
        output = (output * route.scores[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)

        if let sharedExperts {
            output = output + sharedExperts(x)
        }

        return output
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - Model

internal final class NemotronHBlock: Module {
    let kind: NemotronHBlockKind

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: Module

    var attention: NemotronHAttention? {
        mixer as? NemotronHAttention
    }

    init(_ configuration: NemotronHConfiguration, kind: NemotronHBlockKind) {
        self.kind = kind
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )

        switch kind {
        case .mamba:
            _mixer.wrappedValue = NemotronHMambaMixer(configuration)
        case .attention:
            _mixer.wrappedValue = NemotronHAttention(configuration)
        case .feedForward:
            _mixer.wrappedValue = NemotronHDenseMLP(configuration)
        case .routedFeedForward:
            _mixer.wrappedValue = NemotronHMoE(configuration)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        guard let operatorLayer = mixer as? NemotronHBlockOperator else {
            preconditionFailure("NemotronH block mixer does not implement block execution")
        }
        return x + operatorLayer(norm(x), attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
    }
}

internal final class NemotronHBackbone: Module {
    let configuration: NemotronHConfiguration
    let layerPlan: NemotronHLayerPlan

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronHBlock]
    @ModuleInfo(key: "norm_f") var finalNorm: RMSNorm

    init(_ configuration: NemotronHConfiguration) {
        self.configuration = configuration
        self.layerPlan = NemotronHLayerPlan(configuration)

        _embeddings.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize,
            dimensions: configuration.hiddenSize
        )
        _layers.wrappedValue = layerPlan.kinds.map { kind in
            NemotronHBlock(configuration, kind: kind)
        }
        _finalNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embeddings(inputs)
        let attentionMask = attentionMask(for: hiddenStates, cache: cache)
        let ssmMask = ssmMask(for: hiddenStates, cache: cache)

        for (layerIndex, layer) in layers.enumerated() {
            let cache = layerPlan.cacheIndexByLayer[layerIndex].flatMap { cacheIndex in
                cacheValue(in: cache, at: cacheIndex)
            }
            hiddenStates = layer(
                hiddenStates,
                attentionMask: attentionMask,
                ssmMask: ssmMask,
                cache: cache
            )
        }

        return finalNorm(hiddenStates)
    }

    private func attentionMask(
        for hiddenStates: MLXArray,
        cache: [KVCache]?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let index = layerPlan.firstAttentionCacheIndex else {
            return .none
        }
        return createAttentionMask(h: hiddenStates, cache: cacheValue(in: cache, at: index))
    }

    private func ssmMask(for hiddenStates: MLXArray, cache: [KVCache]?) -> MLXArray? {
        guard let index = layerPlan.firstMambaCacheIndex else {
            return nil
        }
        return createSSMMask(h: hiddenStates, cache: cacheValue(in: cache, at: index) as? MambaCache)
    }

    private func cacheValue(in cache: [KVCache]?, at index: Int) -> KVCache? {
        guard let cache, cache.indices.contains(index) else {
            return nil
        }
        return cache[index]
    }
}

internal final class NemotronHModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "backbone") var backbone: NemotronHBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: NemotronHConfiguration

    internal init(_ configuration: NemotronHConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabSize
        self.kvHeads = NemotronHLayerPlan(configuration).kvHeads

        _backbone.wrappedValue = NemotronHBackbone(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabSize,
                bias: false
            )
        }
        super.init()
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: backbone(inputs, cache: cache))
    }

    internal func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        backbone(inputs, cache: cache)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hidden = lastTokenHiddenState(backbone(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hidden), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        NemotronHLayerPlan(configuration).cacheKinds.map { cacheKind in
            switch cacheKind {
            case .mamba:
                MambaCache()
            case .attention:
                KVCacheSimple()
            }
        }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        NemotronHSanitizerPlan(configuration).sanitize(weights)
    }

    internal var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("A_log")
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return backbone.embeddings.asLinear(hiddenStates)
    }
}

extension NemotronHModel: LoRAModel {
    internal var loraLayers: [Module] {
        backbone.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        backbone.layers.compactMap { layer in
            guard let attention = layer.attention else {
                return nil
            }
            return (attention, ["q_proj", "v_proj"])
        }
    }
}

// MARK: - Configuration

internal struct NemotronHConfiguration: Codable, Sendable {
    internal var modelType: String
    internal var vocabSize: Int
    internal var hiddenSize: Int
    internal var numHiddenLayers: Int
    internal var numAttentionHeads: Int
    internal var numKeyValueHeads: Int
    internal var attentionBias: Bool
    internal var mambaNumHeads: Int
    internal var mambaHeadDim: Int
    internal var mambaProjBias: Bool
    internal var ssmStateSize: Int
    internal var convKernel: Int
    internal var nGroups: Int
    internal var intermediateSize: Int
    internal var moeIntermediateSize: Int
    internal var moeSharedExpertIntermediateSize: Int
    internal var nRoutedExperts: Int
    internal var nSharedExperts: Int?
    internal var numExpertsPerTok: Int
    internal var hybridOverridePattern: String
    internal var layerNormEpsilon: Float
    internal var mlpBias: Bool
    internal var useBias: Bool
    internal var useConvBias: Bool
    internal var tieWordEmbeddings: Bool
    internal var ropeTheta: Float
    internal var headDim: Int?
    internal var nGroup: Int
    internal var topkGroup: Int
    internal var normTopkProb: Bool
    internal var routedScalingFactor: Float
    internal var timeStepLimitMin: Float
    internal var timeStepLimitMax: Float

    internal var blockPattern: [NemotronHBlockKind] {
        hybridOverridePattern.compactMap(NemotronHBlockKind.parse)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mambaNumHeads = "mamba_num_heads"
        case mambaHeadDim = "mamba_head_dim"
        case mambaProjBias = "mamba_proj_bias"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridOverridePattern = "hybrid_override_pattern"
        case layerNormEpsilon = "layer_norm_epsilon"
        case mlpBias = "mlp_bias"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case timeStepLimit = "time_step_limit"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
    }

    internal init(
        modelType: String = "nemotron_h",
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        mambaNumHeads: Int,
        mambaHeadDim: Int,
        ssmStateSize: Int,
        convKernel: Int,
        nGroups: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        moeSharedExpertIntermediateSize: Int,
        nRoutedExperts: Int,
        numExpertsPerTok: Int,
        hybridOverridePattern: String,
        layerNormEpsilon: Float = 1e-5,
        attentionBias: Bool = false,
        mambaProjBias: Bool = false,
        mlpBias: Bool = false,
        useBias: Bool = false,
        useConvBias: Bool = true,
        tieWordEmbeddings: Bool = false,
        ropeTheta: Float = 10_000.0,
        headDim: Int? = nil,
        nSharedExperts: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1.0,
        timeStepLimitMin: Float = 0.0,
        timeStepLimitMax: Float = .infinity
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.attentionBias = attentionBias
        self.mambaNumHeads = mambaNumHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = mambaProjBias
        self.ssmStateSize = ssmStateSize
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.hybridOverridePattern = hybridOverridePattern
        self.layerNormEpsilon = layerNormEpsilon
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.headDim = headDim
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        let pattern = try Self.decodePattern(from: container, hiddenLayers: hiddenLayers)
        let limits = try Self.decodeTimeStepLimits(from: container)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "nemotron_h",
            vocabSize: try container.decode(Int.self, forKey: .vocabSize),
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            numHiddenLayers: hiddenLayers,
            numAttentionHeads: try container.decode(Int.self, forKey: .numAttentionHeads),
            numKeyValueHeads: try container.decode(Int.self, forKey: .numKeyValueHeads),
            mambaNumHeads: try container.decode(Int.self, forKey: .mambaNumHeads),
            mambaHeadDim: try container.decode(Int.self, forKey: .mambaHeadDim),
            ssmStateSize: try container.decode(Int.self, forKey: .ssmStateSize),
            convKernel: try container.decode(Int.self, forKey: .convKernel),
            nGroups: try container.decode(Int.self, forKey: .nGroups),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            moeIntermediateSize: try container.decode(Int.self, forKey: .moeIntermediateSize),
            moeSharedExpertIntermediateSize: try container.decode(
                Int.self,
                forKey: .moeSharedExpertIntermediateSize
            ),
            nRoutedExperts: try container.decode(Int.self, forKey: .nRoutedExperts),
            numExpertsPerTok: try container.decode(Int.self, forKey: .numExpertsPerTok),
            hybridOverridePattern: pattern,
            layerNormEpsilon: try container.decodeIfPresent(
                Float.self,
                forKey: .layerNormEpsilon
            ) ?? 1e-5,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            mambaProjBias: try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias)
                ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            useBias: try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            useConvBias: try container.decodeIfPresent(Bool.self, forKey: .useConvBias)
                ?? true,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000.0,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            nSharedExperts: try container.decodeIfPresent(Int.self, forKey: .nSharedExperts),
            nGroup: try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1,
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? true,
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 1.0,
            timeStepLimitMin: limits.minimum,
            timeStepLimitMax: limits.maximum
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(mambaNumHeads, forKey: .mambaNumHeads)
        try container.encode(mambaHeadDim, forKey: .mambaHeadDim)
        try container.encode(mambaProjBias, forKey: .mambaProjBias)
        try container.encode(ssmStateSize, forKey: .ssmStateSize)
        try container.encode(convKernel, forKey: .convKernel)
        try container.encode(nGroups, forKey: .nGroups)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(
            moeSharedExpertIntermediateSize,
            forKey: .moeSharedExpertIntermediateSize
        )
        try container.encode(nRoutedExperts, forKey: .nRoutedExperts)
        try container.encodeIfPresent(nSharedExperts, forKey: .nSharedExperts)
        try container.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try container.encode(hybridOverridePattern, forKey: .hybridOverridePattern)
        try container.encode(layerNormEpsilon, forKey: .layerNormEpsilon)
        try container.encode(mlpBias, forKey: .mlpBias)
        try container.encode(useBias, forKey: .useBias)
        try container.encode(useConvBias, forKey: .useConvBias)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encodeIfPresent(headDim, forKey: .headDim)
        try container.encode(nGroup, forKey: .nGroup)
        try container.encode(topkGroup, forKey: .topkGroup)
        try container.encode(normTopkProb, forKey: .normTopkProb)
        try container.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try container.encode(timeStepLimitMin, forKey: .timeStepLimitMin)
        if timeStepLimitMax.isFinite {
            try container.encode(timeStepLimitMax, forKey: .timeStepLimitMax)
        }
    }

    private static func decodePattern(
        from container: KeyedDecodingContainer<CodingKeys>,
        hiddenLayers: Int
    ) throws -> String {
        let pattern: String
        if let string = try? container.decode(String.self, forKey: .hybridOverridePattern) {
            pattern = string
        } else if let array = try? container.decode(
            [String].self,
            forKey: .hybridOverridePattern
        ) {
            pattern = array.joined()
        } else {
            pattern = String(repeating: NemotronHBlockKind.attention.rawValue, count: hiddenLayers)
        }

        let kinds = pattern.compactMap(NemotronHBlockKind.parse)
        guard kinds.count == pattern.count else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern,
                in: container,
                debugDescription: "hybrid_override_pattern contains an unknown block symbol"
            )
        }
        guard kinds.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern,
                in: container,
                debugDescription: "hybrid_override_pattern must match num_hidden_layers"
            )
        }
        return pattern
    }

    private static func decodeTimeStepLimits(
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> (minimum: Float, maximum: Float) {
        if let values = try container.decodeIfPresent([Float].self, forKey: .timeStepLimit) {
            guard !values.isEmpty else {
                throw DecodingError.dataCorruptedError(
                    forKey: .timeStepLimit,
                    in: container,
                    debugDescription: "time_step_limit cannot be empty"
                )
            }
            return (values[0], values.count > 1 ? values[1] : values[0])
        }

        return (
            try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMin) ?? 0.0,
            try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMax) ?? .infinity
        )
    }
}
