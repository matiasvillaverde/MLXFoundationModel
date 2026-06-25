import Foundation
import MLX
import MLXNN

// MARK: - Plans

internal enum MiMoV2FlashAttentionKind: Equatable, Sendable {
    case full
    case sliding
}

internal struct MiMoV2FlashAttentionLayout: Equatable, Sendable {
    internal let kind: MiMoV2FlashAttentionKind
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let valueHeadDimensions: Int
    internal let rotaryDimensions: Int
    internal let attentionScale: Float
    internal let ropeTheta: Float
    internal let usesAttentionSink: Bool

    internal init(_ config: MiMoV2FlashConfiguration, kind: MiMoV2FlashAttentionKind) {
        let attentionHeads: Int
        let keyValueHeads: Int
        let headDimensions: Int
        let valueHeadDimensions: Int
        let ropeTheta: Float
        let usesAttentionSink: Bool

        switch kind {
        case .full:
            attentionHeads = config.attentionHeads
            keyValueHeads = config.kvHeads
            headDimensions = config.headDim
            valueHeadDimensions = config.vHeadDim
            ropeTheta = config.ropeTheta
            usesAttentionSink = config.addFullAttentionSinkBias
        case .sliding:
            attentionHeads = config.swaAttentionHeads
            keyValueHeads = config.swaKvHeads
            headDimensions = config.swaHeadDim
            valueHeadDimensions = config.swaVHeadDim
            ropeTheta = config.swaRopeTheta
            usesAttentionSink = config.addSwaAttentionSinkBias
        }

        precondition(attentionHeads > 0, "MiMo v2 Flash attention heads must be positive")
        precondition(keyValueHeads > 0, "MiMo v2 Flash KV heads must be positive")
        precondition(headDimensions > 0, "MiMo v2 Flash head dimensions must be positive")
        precondition(
            attentionHeads.isMultiple(of: keyValueHeads),
            "MiMo v2 Flash attention heads must group KV heads"
        )
        precondition(
            config.partialRotaryFactor > 0,
            "MiMo v2 Flash partial rotary factor must be positive"
        )

        self.kind = kind
        self.hiddenSize = config.hiddenSize
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads
        self.headDimensions = headDimensions
        self.valueHeadDimensions = valueHeadDimensions
        self.rotaryDimensions = max(1, Int(Float(headDimensions) * config.partialRotaryFactor))
        self.attentionScale = pow(Float(headDimensions), -0.5)
        self.ropeTheta = ropeTheta
        self.usesAttentionSink = usesAttentionSink
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyProjectionSize: Int { keyValueHeads * headDimensions }
    internal var valueProjectionSize: Int { keyValueHeads * valueHeadDimensions }
    internal var outputProjectionInputSize: Int { attentionHeads * valueHeadDimensions }
}

internal struct MiMoV2FlashLayerSchedule: Equatable, Sendable {
    internal let hybridPattern: [Int]
    internal let moePattern: [Int]
    internal let slidingWindowSize: Int

    internal init(_ config: MiMoV2FlashConfiguration) {
        precondition(config.hiddenLayers > 0, "MiMo v2 Flash must have at least one layer")
        precondition(
            config.hybridLayerPattern.count == config.hiddenLayers,
            "MiMo v2 Flash hybrid pattern must match layer count"
        )
        precondition(
            config.moeLayerFreq.count == config.hiddenLayers,
            "MiMo v2 Flash MoE pattern must match layer count"
        )
        precondition(
            config.hybridLayerPattern.allSatisfy { $0 == 0 || $0 == 1 },
            "MiMo v2 Flash hybrid pattern entries must be 0 or 1"
        )
        precondition(
            config.moeLayerFreq.allSatisfy { $0 == 0 || $0 == 1 },
            "MiMo v2 Flash MoE pattern entries must be 0 or 1"
        )

        self.hybridPattern = config.hybridLayerPattern
        self.moePattern = config.moeLayerFreq
        self.slidingWindowSize = config.slidingWindowSize
    }

    internal var layerCount: Int {
        hybridPattern.count
    }

    internal var firstFullLayer: Int {
        hybridPattern.firstIndex(of: 0) ?? 0
    }

    internal var firstSlidingLayer: Int {
        hybridPattern.firstIndex(of: 1) ?? firstFullLayer
    }

    internal func attentionKind(layerIndex: Int) -> MiMoV2FlashAttentionKind {
        hybridPattern[layerIndex] == 1 ? .sliding : .full
    }

    internal func usesMoE(layerIndex: Int) -> Bool {
        moePattern[layerIndex] == 1
    }
}

internal struct MiMoV2FlashRoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizesSelectedProbabilities: Bool
    internal let routedScalingFactor: Float

    internal init(_ config: MiMoV2FlashConfiguration) {
        let expertCount = config.nRoutedExperts ?? 0
        precondition(expertCount > 0, "MiMo v2 Flash routed expert count must be positive")
        precondition(
            config.topkMethod == "noaux_tc",
            "MiMo v2 Flash only supports noaux_tc routing"
        )
        precondition(config.numExpertsPerTok > 0, "MiMo v2 Flash top-k must be positive")
        precondition(config.nGroup > 0, "MiMo v2 Flash group count must be positive")
        precondition(
            expertCount.isMultiple(of: config.nGroup),
            "MiMo v2 Flash experts must divide evenly into groups"
        )
        precondition(
            config.numExpertsPerTok <= expertCount,
            "MiMo v2 Flash top-k cannot exceed routed expert count"
        )

        let keptGroupCount = min(config.topkGroup, config.nGroup)
        precondition(
            keptGroupCount > 0 && keptGroupCount <= config.nGroup,
            "MiMo v2 Flash kept group count must be within group count"
        )

        self.expertCount = expertCount
        self.selectedExpertCount = config.numExpertsPerTok
        self.groupCount = config.nGroup
        self.keptGroupCount = keptGroupCount
        self.normalizesSelectedProbabilities = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor ?? 1
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
        let originalScores = sigmoid(logits.asType(.float32))
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

private enum MiMoV2FlashExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal struct MiMoV2FlashExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: MiMoV2FlashConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.nRoutedExperts ?? 0
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in MiMoV2FlashExpertProjection.allCases {
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

private func mimoV2FlashAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }

    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys,
            values: values
        )
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    }

    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: cachedKeys,
        values: cachedValues,
        scale: scale,
        mask: mask,
        sinks: sinks
    )
}

// MARK: - Model Components

internal final class MiMoV2FlashAttention: Module {
    let layout: MiMoV2FlashAttentionLayout

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ParameterInfo(key: "attention_sink_bias") var attentionSinkBias: MLXArray

    let rope: RoPE

    init(_ config: MiMoV2FlashConfiguration, kind: MiMoV2FlashAttentionKind) {
        self.layout = MiMoV2FlashAttentionLayout(config, kind: kind)

        _query.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        _key.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyProjectionSize,
            bias: false
        )
        _value.wrappedValue = Linear(
            layout.hiddenSize,
            layout.valueProjectionSize,
            bias: false
        )
        _output.wrappedValue = Linear(
            layout.outputProjectionInputSize,
            layout.hiddenSize,
            bias: false
        )
        _attentionSinkBias.wrappedValue = MLXArray.ones([layout.attentionHeads])

        self.rope = RoPE(
            dimensions: layout.rotaryDimensions,
            traditional: false,
            base: layout.ropeTheta
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
            .transposed(0, 2, 1, 3)
        var keyStates = key(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let valueStates = value(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.valueHeadDimensions)
            .transposed(0, 2, 1, 3)

        queryStates = applyRotaryPosition(rope, to: queryStates, cache: cache)
        keyStates = applyRotaryPosition(rope, to: keyStates, cache: cache)

        let attentionOutput = mimoV2FlashAttention(
            queries: queryStates,
            keys: keyStates,
            values: valueStates,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask,
            sinks: layout.usesAttentionSink ? attentionSinkBias : nil
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return output(attentionOutput)
    }

    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        if parameter == "attention_sink_bias" {
            return
        }
        try super.updateMissing(
            parameter: parameter,
            verify: verify,
            path: path,
            modulePath: modulePath
        )
    }
}

internal final class MiMoV2FlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: MiMoV2FlashConfiguration, intermediateSize: Int? = nil) {
        let intermediateSize = intermediateSize ?? config.intermediateSize
        _gate.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

internal final class MiMoV2FlashGate: Module {
    let routingPlan: MiMoV2FlashRoutingPlan

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: MiMoV2FlashConfiguration) {
        self.routingPlan = MiMoV2FlashRoutingPlan(config)
        _weight.wrappedValue = MLXArray.zeros([routingPlan.expertCount, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([routingPlan.expertCount])
    }

    func callAsFunction(_ x: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        routingPlan.route(
            logits: x.matmul(weight.T),
            correctionBias: eScoreCorrectionBias,
            outputDType: x.dtype
        )
    }
}

internal final class MiMoV2FlashMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: MiMoV2FlashGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiMoV2FlashMLP?

    init(_ config: MiMoV2FlashConfiguration) {
        let routingPlan = MiMoV2FlashRoutingPlan(config)

        _gate.wrappedValue = MiMoV2FlashGate(config)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: routingPlan.expertCount
        )

        if let sharedExpertCount = config.nSharedExperts, sharedExpertCount > 0 {
            _sharedExperts.wrappedValue = MiMoV2FlashMLP(
                config,
                intermediateSize: config.moeIntermediateSize * sharedExpertCount
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = gate(x)
        let expertOutput = switchMLP(x, routed.indices)
        var output = (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            output = output + sharedExperts(x)
        }
        return output
    }
}

internal final class MiMoV2FlashDecoderLayer: Module {
    let attentionKind: MiMoV2FlashAttentionKind

    @ModuleInfo(key: "self_attn") fileprivate var attention: MiMoV2FlashAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(
        _ config: MiMoV2FlashConfiguration,
        attentionKind: MiMoV2FlashAttentionKind,
        usesMoE: Bool
    ) {
        self.attentionKind = attentionKind
        _attention.wrappedValue = MiMoV2FlashAttention(config, kind: attentionKind)
        if usesMoE {
            _feedForward.wrappedValue = MiMoV2FlashMoE(config)
        } else {
            _feedForward.wrappedValue = MiMoV2FlashMLP(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layernormEpsilon
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layernormEpsilon
        )
    }

    var usesSlidingWindow: Bool {
        attentionKind == .sliding
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

internal final class MiMoV2FlashBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [MiMoV2FlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private let schedule: MiMoV2FlashLayerSchedule

    init(_ config: MiMoV2FlashConfiguration) {
        let schedule = MiMoV2FlashLayerSchedule(config)
        self.schedule = schedule

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< schedule.layerCount).map { layerIndex in
            MiMoV2FlashDecoderLayer(
                config,
                attentionKind: schedule.attentionKind(layerIndex: layerIndex),
                usesMoE: schedule.usesMoE(layerIndex: layerIndex)
            )
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layernormEpsilon
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        let fullMask = createAttentionMask(h: hidden, cache: cache?[schedule.firstFullLayer])
        let slidingMask = createAttentionMask(
            h: hidden,
            cache: cache?[schedule.firstSlidingLayer],
            windowSize: schedule.slidingWindowSize
        )

        for (layerIndex, layer) in layers.enumerated() {
            let mask = layer.usesSlidingWindow ? slidingMask : fullMask
            hidden = layer(hidden, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hidden)
    }
}

internal final class MiMoV2FlashModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let modelType: String
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: MiMoV2FlashBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    private let configuration: MiMoV2FlashConfiguration
    private let schedule: MiMoV2FlashLayerSchedule

    internal init(_ config: MiMoV2FlashConfiguration) {
        let schedule = MiMoV2FlashLayerSchedule(config)

        self.configuration = config
        self.schedule = schedule
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { layerIndex in
            schedule.attentionKind(layerIndex: layerIndex) == .sliding
                ? config.swaKvHeads
                : config.kvHeads
        }
        _model.wrappedValue = MiMoV2FlashBackbone(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: lmHead(hiddenStates), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { layerIndex in
            if schedule.attentionKind(layerIndex: layerIndex) == .sliding {
                RotatingKVCache(maxSize: configuration.slidingWindowSize)
            } else {
                KVCacheSimple()
            }
        }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .block(),
            sidecarPolicy: .dropActivationScale
        ).weights
        return MiMoV2FlashExpertPackingPlan(configuration)
            .pack(sanitized)
            .filter { key, _ in !key.hasPrefix("model.mtp") }
    }
}

// MARK: - Configuration

internal struct MiMoV2FlashConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var numExpertsPerTok: Int
    var hybridLayerPattern: [Int]
    var moeLayerFreq: [Int]
    var addSwaAttentionSinkBias: Bool
    var addFullAttentionSinkBias: Bool
    var slidingWindowSize: Int
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float?
    var topkMethod: String
    var scoringFunc: String
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var maxPositionEmbeddings: Int
    var layernormEpsilon: Float
    var ropeTheta: Float
    var swaRopeTheta: Float
    var swaAttentionHeads: Int
    var swaKvHeads: Int
    var headDim: Int
    var vHeadDim: Int
    var swaHeadDim: Int
    var swaVHeadDim: Int
    var partialRotaryFactor: Float

    internal init(
        modelType: String = "mimo_v2_flash",
        numExpertsPerTok: Int = 1,
        hybridLayerPattern: [Int],
        moeLayerFreq: [Int],
        addSwaAttentionSinkBias: Bool = false,
        addFullAttentionSinkBias: Bool = false,
        slidingWindowSize: Int,
        vocabularySize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int? = nil,
        hiddenLayers: Int,
        attentionHeads: Int,
        kvHeads: Int,
        nSharedExperts: Int? = nil,
        nRoutedExperts: Int? = nil,
        routedScalingFactor: Float? = nil,
        topkMethod: String = "noaux_tc",
        scoringFunc: String = "sigmoid",
        normTopkProb: Bool = false,
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        maxPositionEmbeddings: Int = 131_072,
        layernormEpsilon: Float = 1e-6,
        ropeTheta: Float = 10_000,
        swaRopeTheta: Float? = nil,
        headDim: Int? = nil,
        vHeadDim: Int? = nil,
        swaAttentionHeads: Int? = nil,
        swaKvHeads: Int? = nil,
        swaHeadDim: Int? = nil,
        swaVHeadDim: Int? = nil,
        partialRotaryFactor: Float = 1
    ) {
        self.modelType = modelType
        self.numExpertsPerTok = numExpertsPerTok
        self.hybridLayerPattern = hybridLayerPattern
        self.moeLayerFreq = moeLayerFreq
        self.addSwaAttentionSinkBias = addSwaAttentionSinkBias
        self.addFullAttentionSinkBias = addFullAttentionSinkBias
        self.slidingWindowSize = slidingWindowSize
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize ?? intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.nSharedExperts = nSharedExperts
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = routedScalingFactor
        self.topkMethod = topkMethod
        self.scoringFunc = scoringFunc
        self.normTopkProb = normTopkProb
        self.nGroup = nGroup
        self.topkGroup = topkGroup ?? nGroup
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.layernormEpsilon = layernormEpsilon
        self.ropeTheta = ropeTheta
        self.swaRopeTheta = swaRopeTheta ?? ropeTheta
        self.swaAttentionHeads = swaAttentionHeads ?? attentionHeads
        self.swaKvHeads = swaKvHeads ?? kvHeads
        let defaultHeadDim = hiddenSize / attentionHeads
        self.headDim = headDim ?? defaultHeadDim
        self.vHeadDim = vHeadDim ?? defaultHeadDim
        self.swaHeadDim = swaHeadDim ?? self.headDim
        self.swaVHeadDim = swaVHeadDim ?? self.vHeadDim
        self.partialRotaryFactor = partialRotaryFactor
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridLayerPattern = "hybrid_layer_pattern"
        case moeLayerFreq = "moe_layer_freq"
        case addSwaAttentionSinkBias = "add_swa_attention_sink_bias"
        case addFullAttentionSinkBias = "add_full_attention_sink_bias"
        case slidingWindowSize = "sliding_window_size"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layernormEpsilon = "layernorm_epsilon"
        case ropeTheta = "rope_theta"
        case swaRopeTheta = "swa_rope_theta"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKvHeads = "swa_num_key_value_heads"
        case headDim = "head_dim"
        case vHeadDim = "v_head_dim"
        case swaHeadDim = "swa_head_dim"
        case swaVHeadDim = "swa_v_head_dim"
        case partialRotaryFactor = "partial_rotary_factor"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        let hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        let nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "mimo_v2_flash",
            numExpertsPerTok: try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
                ?? 1,
            hybridLayerPattern: try container.decodeIfPresent(
                [Int].self,
                forKey: .hybridLayerPattern
            ) ?? Array(repeating: 0, count: hiddenLayers),
            moeLayerFreq: try container.decodeIfPresent([Int].self, forKey: .moeLayerFreq)
                ?? Array(repeating: 0, count: hiddenLayers),
            addSwaAttentionSinkBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .addSwaAttentionSinkBias
            ) ?? false,
            addFullAttentionSinkBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .addFullAttentionSinkBias
            ) ?? false,
            slidingWindowSize: try container.decodeIfPresent(Int.self, forKey: .slidingWindowSize)
                ?? 4_096,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: hiddenSize,
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ),
            hiddenLayers: hiddenLayers,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            nSharedExperts: try container.decodeIfPresent(Int.self, forKey: .nSharedExperts),
            nRoutedExperts: try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts),
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ),
            topkMethod: try container.decodeIfPresent(String.self, forKey: .topkMethod)
                ?? "noaux_tc",
            scoringFunc: try container.decodeIfPresent(String.self, forKey: .scoringFunc)
                ?? "sigmoid",
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false,
            nGroup: nGroup,
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? nGroup,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            layernormEpsilon: try container.decodeIfPresent(
                Float.self,
                forKey: .layernormEpsilon
            ) ?? 1e-6,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            swaRopeTheta: try container.decodeIfPresent(Float.self, forKey: .swaRopeTheta),
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim)
                ?? (hiddenSize / attentionHeads),
            vHeadDim: try container.decodeIfPresent(Int.self, forKey: .vHeadDim),
            swaAttentionHeads: try container.decodeIfPresent(Int.self, forKey: .swaAttentionHeads),
            swaKvHeads: try container.decodeIfPresent(Int.self, forKey: .swaKvHeads),
            swaHeadDim: try container.decodeIfPresent(Int.self, forKey: .swaHeadDim),
            swaVHeadDim: try container.decodeIfPresent(Int.self, forKey: .swaVHeadDim),
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 1
        )
    }
}

// MARK: - LoRA

extension MiMoV2FlashModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
