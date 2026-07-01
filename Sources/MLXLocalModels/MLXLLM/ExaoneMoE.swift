import Foundation
import MLX
import MLXNN

internal struct ExaoneMoEConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var layerTypes: [String]
    var usesMoELayer: [Bool]
    var numExperts: Int
    var numExpertsPerToken: Int
    var numSharedExperts: Int
    var nGroup: Int
    var topkGroup: Int
    var routedScalingFactor: Float
    var normTopkProb: Bool
    var scoringFunc: String
    var topkMethod: String
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var ropeParameters: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "exaone_moe",
        vocabularySize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        hiddenLayers: Int,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        rmsNormEps: Float = 1e-5,
        maxPositionEmbeddings: Int = 131_072,
        slidingWindow: Int = 128,
        layerTypes: [String]? = nil,
        usesMoELayer: [Bool]? = nil,
        numExperts: Int,
        numExpertsPerToken: Int,
        numSharedExperts: Int = 1,
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        routedScalingFactor: Float = 2.5,
        normTopkProb: Bool = true,
        scoringFunc: String = "sigmoid",
        topkMethod: String = "noaux_tc",
        ropeTheta: Float = 1_000_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        ropeParameters: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false
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
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.slidingWindow = slidingWindow
        self.layerTypes = Self.resolvedLayerTypes(layerTypes, count: hiddenLayers)
        self.usesMoELayer = Self.resolvedMoELayers(usesMoELayer, count: hiddenLayers)
        self.numExperts = numExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.numSharedExperts = numSharedExperts
        self.nGroup = nGroup
        self.topkGroup = topkGroup ?? nGroup
        self.routedScalingFactor = routedScalingFactor
        self.normTopkProb = normTopkProb
        self.scoringFunc = scoringFunc
        self.topkMethod = topkMethod
        self.ropeParameters = ropeParameters
        self.ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? ropeTheta
        self.ropeScaling = ropeScaling ?? ropeParameters
        self.tieWordEmbeddings = tieWordEmbeddings
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
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case layerTypes = "layer_types"
        case usesMoELayer = "is_moe_layer"
        case mlpLayerTypes = "mlp_layer_types"
        case firstKDenseReplace = "first_k_dense_replace"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        let layerTypes = try Self.decodeLayerTypes(
            from: container,
            hiddenLayers: hiddenLayers
        )
        let usesMoELayer = try Self.decodeMoELayers(from: container, hiddenLayers: hiddenLayers)
        let ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeParameters
        )

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "exaone_moe",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: hiddenSize,
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            moeIntermediateSize: try container.decode(Int.self, forKey: .moeIntermediateSize),
            hiddenLayers: hiddenLayers,
            attentionHeads: attentionHeads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim)
                ?? hiddenSize / attentionHeads,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
                ?? 128,
            layerTypes: layerTypes,
            usesMoELayer: usesMoELayer,
            numExperts: try container.decode(Int.self, forKey: .numExperts),
            numExpertsPerToken: try container.decode(Int.self, forKey: .numExpertsPerToken),
            numSharedExperts: try container.decodeIfPresent(Int.self, forKey: .numSharedExperts)
                ?? 1,
            nGroup: nGroup,
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup)
                ?? nGroup,
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 2.5,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? true,
            scoringFunc: try container.decodeIfPresent(String.self, forKey: .scoringFunc)
                ?? "sigmoid",
            topkMethod: try container.decodeIfPresent(String.self, forKey: .topkMethod)
                ?? "noaux_tc",
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 1_000_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            ropeParameters: ropeParameters,
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? false
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: Self.CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encode(usesMoELayer, forKey: .usesMoELayer)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try container.encode(numSharedExperts, forKey: .numSharedExperts)
        try container.encode(nGroup, forKey: .nGroup)
        try container.encode(topkGroup, forKey: .topkGroup)
        try container.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try container.encode(normTopkProb, forKey: .normTopkProb)
        try container.encode(scoringFunc, forKey: .scoringFunc)
        try container.encode(topkMethod, forKey: .topkMethod)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encodeIfPresent(ropeParameters, forKey: .ropeParameters)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }

    private static func decodeLayerTypes(
        from container: KeyedDecodingContainer<CodingKeys>,
        hiddenLayers: Int
    ) throws -> [String]? {
        if let layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            return layerTypes
        }
        guard let pattern = try container.decodeIfPresent(String.self, forKey: .slidingWindowPattern)
        else {
            return nil
        }
        let patternTypes = pattern.map { character in
            character == "L" ? "sliding_attention" : "full_attention"
        }
        return resolvedLayerTypes(patternTypes, count: hiddenLayers)
    }

    private static func decodeMoELayers(
        from container: KeyedDecodingContainer<CodingKeys>,
        hiddenLayers: Int
    ) throws -> [Bool]? {
        if let flags = try container.decodeIfPresent([Bool].self, forKey: .usesMoELayer) {
            return flags
        }
        if let mlpTypes = try container.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            return resolvedMoELayers(
                mlpTypes.map { type in
                    type != "dense" && type != "mlp"
                },
                count: hiddenLayers
            )
        }
        if let densePrefix = try container.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) {
            return (0 ..< hiddenLayers).map { $0 >= densePrefix }
        }
        return nil
    }

    private static func resolvedLayerTypes(_ values: [String]?, count: Int) -> [String] {
        guard let values, !values.isEmpty else {
            return Array(repeating: "full_attention", count: count)
        }
        if values.count == count {
            return values
        }
        return (0 ..< count).map { values[$0 % values.count] }
    }

    private static func resolvedMoELayers(_ values: [Bool]?, count: Int) -> [Bool] {
        guard let values, !values.isEmpty else {
            return Array(repeating: true, count: count)
        }
        if values.count == count {
            return values
        }
        return (0 ..< count).map { values[$0 % values.count] }
    }
}

internal enum ExaoneMoEAttentionKind: String, Equatable, Sendable {
    case full = "full_attention"
    case sliding = "sliding_attention"

    internal init(_ value: String) {
        self = ExaoneMoEAttentionKind(rawValue: value) ?? .full
    }
}

internal struct ExaoneMoELayerPlan: Equatable, Sendable {
    internal let attentionKinds: [ExaoneMoEAttentionKind]
    internal let usesMoELayer: [Bool]
    internal let slidingWindow: Int

    internal init(_ config: ExaoneMoEConfiguration) {
        precondition(config.hiddenLayers > 0, "EXAONE MoE must have at least one layer")
        precondition(
            config.layerTypes.count == config.hiddenLayers,
            "EXAONE MoE layer_types must match hidden layer count"
        )
        precondition(
            config.usesMoELayer.count == config.hiddenLayers,
            "EXAONE MoE MLP layer plan must match hidden layer count"
        )
        precondition(config.slidingWindow > 0, "EXAONE MoE sliding window must be positive")

        self.attentionKinds = config.layerTypes.map(ExaoneMoEAttentionKind.init)
        self.usesMoELayer = config.usesMoELayer
        self.slidingWindow = config.slidingWindow
    }

    internal var layerCount: Int { attentionKinds.count }
    internal var hasSlidingAttention: Bool { attentionKinds.contains(.sliding) }
    internal var firstFullLayer: Int { attentionKinds.firstIndex(of: .full) ?? 0 }
    internal var firstSlidingLayer: Int? { attentionKinds.firstIndex(of: .sliding) }

    internal func attentionKind(layerIndex: Int) -> ExaoneMoEAttentionKind {
        attentionKinds[layerIndex]
    }

    internal func usesSparseExperts(layerIndex: Int) -> Bool {
        usesMoELayer[layerIndex]
    }
}

internal struct ExaoneMoEAttentionLayout: Equatable, Sendable {
    internal let kind: ExaoneMoEAttentionKind
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let attentionScale: Float
    internal let usesRotaryPosition: Bool

    internal init(
        _ config: ExaoneMoEConfiguration,
        kind: ExaoneMoEAttentionKind,
        applyRopeAllLayers: Bool
    ) {
        precondition(config.hiddenSize > 0, "EXAONE MoE hidden size must be positive")
        precondition(config.attentionHeads > 0, "EXAONE MoE attention heads must be positive")
        precondition(config.kvHeads > 0, "EXAONE MoE KV heads must be positive")
        precondition(config.headDim > 0, "EXAONE MoE head dimension must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "EXAONE MoE attention heads must group KV heads"
        )

        self.kind = kind
        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = config.headDim
        self.attentionScale = pow(Float(config.headDim), -0.5)
        self.usesRotaryPosition = kind == .sliding || applyRopeAllLayers
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
}

internal enum ExaoneMoEScoreFunction: String, Equatable, Sendable {
    case sigmoid
    case softmax

    internal init(_ value: String) {
        self = ExaoneMoEScoreFunction(rawValue: value) ?? .sigmoid
    }

    internal func scores(from logits: MLXArray) -> MLXArray {
        switch self {
        case .sigmoid:
            MLX.sigmoid(logits.asType(.float32))
        case .softmax:
            MLX.softmax(logits.asType(.float32), axis: -1, precise: true)
        }
    }
}

internal struct ExaoneMoERoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizesSelectedProbabilities: Bool
    internal let routedScalingFactor: Float
    internal let scoreFunction: ExaoneMoEScoreFunction

    internal init(_ config: ExaoneMoEConfiguration) {
        let keptGroupCount = min(config.topkGroup, config.nGroup)

        precondition(config.topkMethod == "noaux_tc", "EXAONE MoE only supports noaux_tc routing")
        precondition(config.numExperts > 0, "EXAONE MoE expert count must be positive")
        precondition(config.numExpertsPerToken > 0, "EXAONE MoE top-k must be positive")
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "EXAONE MoE top-k cannot exceed expert count"
        )
        precondition(config.nGroup > 0, "EXAONE MoE group count must be positive")
        precondition(
            config.numExperts.isMultiple(of: config.nGroup),
            "EXAONE MoE experts must divide evenly into groups"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= config.nGroup,
            "EXAONE MoE kept-group count must be within group count"
        )

        self.expertCount = config.numExperts
        self.selectedExpertCount = config.numExpertsPerToken
        self.groupCount = config.nGroup
        self.keptGroupCount = keptGroupCount
        self.normalizesSelectedProbabilities = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.scoreFunction = ExaoneMoEScoreFunction(config.scoringFunc)
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
    ) -> (indices: MLXArray, scores: MLXArray) {
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

        return (indices, (selectedScores * routedScalingFactor).asType(outputDType))
    }
}

private enum ExaoneMoEExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal struct ExaoneMoEExpertPackingPlan: Equatable, Sendable {
    internal let layerPlan: ExaoneMoELayerPlan
    internal let expertCount: Int

    internal init(_ config: ExaoneMoEConfiguration) {
        self.layerPlan = ExaoneMoELayerPlan(config)
        self.expertCount = config.numExperts
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights

        for layerIndex in 0 ..< layerPlan.layerCount where layerPlan.usesSparseExperts(
            layerIndex: layerIndex
        ) {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in ExaoneMoEExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let keys = (0 ..< expertCount).map { expertIndex in
                        "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                    }
                    let tensors = keys.compactMap { packed[$0] }
                    guard tensors.count == expertCount else {
                        continue
                    }

                    for key in keys {
                        packed[key] = nil
                    }
                    packed["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        stacked(tensors)
                }
            }
        }

        return packed
    }
}

private struct ExaoneMoEWeightSanitizer {
    let config: ExaoneMoEConfiguration

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        sanitized = sanitized.filter { key, _ in
            !key.hasPrefix("mtp.") && !key.contains("rotary_emb.inv_freq")
        }

        let layerPlan = ExaoneMoELayerPlan(config)
        for layerIndex in 0 ..< layerPlan.layerCount where layerPlan.usesSparseExperts(
            layerIndex: layerIndex
        ) {
            let prefix = "model.layers.\(layerIndex).mlp"
            if let bias = sanitized.removeValue(forKey: "\(prefix).e_score_correction_bias") {
                sanitized["\(prefix).gate.e_score_correction_bias"] = bias
            }
        }

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }

        return ExaoneMoEExpertPackingPlan(config).pack(sanitized)
    }
}

private final class ExaoneMoEAttention: Module {
    let layout: ExaoneMoEAttentionLayout
    private let rope: RoPELayer?

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") private var keyNorm: RMSNorm

    init(
        _ config: ExaoneMoEConfiguration,
        kind: ExaoneMoEAttentionKind,
        applyRopeAllLayers: Bool
    ) {
        self.layout = ExaoneMoEAttentionLayout(
            config,
            kind: kind,
            applyRopeAllLayers: applyRopeAllLayers
        )

        self._queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        self._keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: false
        )
        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: layout.headDimensions,
            eps: config.rmsNormEps
        )
        self._keyNorm.wrappedValue = RMSNorm(
            dimensions: layout.headDimensions,
            eps: config.rmsNormEps
        )

        if layout.usesRotaryPosition {
            self.rope = initializeRope(
                dims: layout.headDimensions,
                base: config.ropeTheta,
                traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings
            )
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)

        var queries = queryProjection(input)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        if let rope {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        return outputProjection(
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.attentionScale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, layout.queryProjectionSize)
        )
    }
}

private final class ExaoneMoEDenseFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    convenience init(_ config: ExaoneMoEConfiguration, intermediateSize: Int? = nil) {
        self.init(
            hiddenSize: config.hiddenSize,
            intermediateSize: intermediateSize ?? config.intermediateSize
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class ExaoneMoEGate: Module {
    private let routingPlan: ExaoneMoERoutingPlan

    @ParameterInfo(key: "weight") private var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") private var correctionBias: MLXArray

    init(_ config: ExaoneMoEConfiguration) {
        self.routingPlan = ExaoneMoERoutingPlan(config)
        self._weight.wrappedValue = zeros([routingPlan.expertCount, config.hiddenSize])
        self._correctionBias.wrappedValue = zeros([routingPlan.expertCount])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        routingPlan.route(
            logits: input.matmul(weight.T),
            correctionBias: correctionBias,
            outputDType: input.dtype
        )
    }
}

private final class ExaoneMoESparseFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") private var gate: ExaoneMoEGate
    @ModuleInfo(key: "shared_experts") private var sharedExperts: ExaoneMoEDenseFeedForward?

    init(_ config: ExaoneMoEConfiguration) {
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        self._gate.wrappedValue = ExaoneMoEGate(config)

        if config.numSharedExperts > 0 {
            self._sharedExperts.wrappedValue = ExaoneMoEDenseFeedForward(
                config,
                intermediateSize: config.moeIntermediateSize * config.numSharedExperts
            )
        }

        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let route = gate(input)
        var output = switchMLP(input, route.indices)
        output = (output * route.scores[.ellipsis, .newAxis])
            .sum(axis: -2)
            .asType(output.dtype)
        if let sharedExperts {
            output = output + sharedExperts(input)
        }
        return output
    }
}

private final class ExaoneMoEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: ExaoneMoEAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: ExaoneMoEConfiguration, layerIndex: Int, layerPlan: ExaoneMoELayerPlan) {
        self._attention.wrappedValue = ExaoneMoEAttention(
            config,
            kind: layerPlan.attentionKind(layerIndex: layerIndex),
            applyRopeAllLayers: !layerPlan.hasSlidingAttention
        )

        if layerPlan.usesSparseExperts(layerIndex: layerIndex) {
            self._feedForward.wrappedValue = ExaoneMoESparseFeedForward(config)
        } else {
            self._feedForward.wrappedValue = ExaoneMoEDenseFeedForward(config)
        }
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
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = attention(
            inputLayerNorm(input),
            mask: mask,
            cache: cache
        )
        let hidden = input + attentionOutput
        return hidden + feedForward(postAttentionLayerNorm(hidden))
    }
}

private final class ExaoneMoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    @ModuleInfo fileprivate var layers: [ExaoneMoEDecoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    private let layerPlan: ExaoneMoELayerPlan

    init(_ config: ExaoneMoEConfiguration) {
        precondition(config.vocabularySize > 0, "EXAONE MoE vocabulary size must be positive")

        let layerPlan = ExaoneMoELayerPlan(config)
        self.layerPlan = layerPlan

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< layerPlan.layerCount).map { layerIndex in
            ExaoneMoEDecoderLayer(config, layerIndex: layerIndex, layerPlan: layerPlan)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        let fullMask = createAttentionMask(h: hidden, cache: cache?[layerPlan.firstFullLayer])
        let slidingMask = layerPlan.firstSlidingLayer.map { layerIndex in
            createAttentionMask(
                h: hidden,
                cache: cache?[layerIndex],
                windowSize: layerPlan.slidingWindow
            )
        }

        for (layerIndex, layer) in layers.enumerated() {
            let kind = layerPlan.attentionKind(layerIndex: layerIndex)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                kind == .sliding ? (slidingMask ?? fullMask) : fullMask
            hidden = layer(hidden, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hidden)
    }
}

internal final class ExaoneMoEModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let configuration: ExaoneMoEConfiguration
    private let layerPlan: ExaoneMoELayerPlan

    @ModuleInfo(key: "model") private var model: ExaoneMoEBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: ExaoneMoEConfiguration) {
        self.configuration = config
        self.layerPlan = ExaoneMoELayerPlan(config)
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)

        self._model.wrappedValue = ExaoneMoEBackbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var output = model(inputs, cache: cache)
        if let lmHead {
            output = lmHead(output)
        } else {
            output = model.embedTokens.asLinear(output)
        }
        return output
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        var logits = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        if let lmHead {
            logits = lmHead(logits)
        } else {
            logits = model.embedTokens.asLinear(logits)
        }
        return greedyTokenOutput(logits: logits, state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        ExaoneMoEWeightSanitizer(config: configuration).sanitize(weights)
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        layerPlan.attentionKinds.map { kind in
            if kind == .sliding {
                return RotatingKVCache(maxSize: layerPlan.slidingWindow, keep: 0)
            }
            return StandardKVCache()
        }
    }
}

extension ExaoneMoEModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
