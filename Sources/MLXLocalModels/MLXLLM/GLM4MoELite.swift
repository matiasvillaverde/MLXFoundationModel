import Foundation
import MLX
import MLXNN

// MARK: - Planning

internal struct GLM4MoELiteAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let kvLoraRank: Int
    internal let qLoraRank: Int?
    internal let qkRopeHeadDim: Int
    internal let qkNopeHeadDim: Int
    internal let valueHeadDim: Int
    internal let queryHeadDim: Int
    internal let attentionScale: Float

    internal init(_ config: GLM4MoELiteConfiguration) {
        precondition(config.hiddenSize > 0, "GLM4 MoE Lite hidden size must be positive")
        precondition(config.attentionHeads > 0, "GLM4 MoE Lite attention heads must be positive")
        precondition(config.kvLoraRank > 0, "GLM4 MoE Lite KV LoRA rank must be positive")
        precondition(config.qkRopeHeadDim > 0, "GLM4 MoE Lite rope head dim must be positive")
        precondition(config.qkNopeHeadDim > 0, "GLM4 MoE Lite non-rope head dim must be positive")
        precondition(config.vHeadDim > 0, "GLM4 MoE Lite value head dim must be positive")

        let queryHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        var scale = pow(Float(queryHeadDim), -0.5)
        if let ropeScaling = config.ropeScaling,
            let allDimScale = ropeScaling["mscale_all_dim"]?.asFloat(),
            let factor = ropeScaling["factor"]?.asFloat(),
            allDimScale != 0,
            factor > 1
        {
            let correction = 0.1 * allDimScale * log(factor) + 1.0
            scale *= correction * correction
        }

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.kvLoraRank = config.kvLoraRank
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.valueHeadDim = config.vHeadDim
        self.queryHeadDim = queryHeadDim
        self.attentionScale = scale
    }

    internal var queryProjectionDimensions: Int {
        attentionHeads * queryHeadDim
    }

    internal var compressedKeyValueDimensions: Int {
        kvLoraRank + qkRopeHeadDim
    }

    internal var outputProjectionDimensions: Int {
        attentionHeads * valueHeadDim
    }

    internal static func ropeScalingForRuntime(
        _ ropeScaling: [String: StringOrNumber]?
    ) -> [String: StringOrNumber]? {
        guard let ropeScaling else { return nil }
        let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"]
        guard case .string("deepseek_yarn") = ropeType else {
            return ropeScaling
        }
        var updated = ropeScaling
        updated["type"] = .string("yarn")
        return updated
    }
}

internal struct GLM4MoELiteLayerPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let firstSparseLayer: Int
    internal let moeLayerFrequency: Int
    internal let routedExpertCount: Int?

    internal init(_ config: GLM4MoELiteConfiguration) {
        precondition(config.hiddenLayers > 0, "GLM4 MoE Lite must have at least one layer")
        precondition(config.firstKDenseReplace >= 0, "GLM4 MoE Lite dense layer count is negative")
        precondition(
            config.firstKDenseReplace <= config.hiddenLayers,
            "GLM4 MoE Lite dense layer count exceeds layer count"
        )
        precondition(config.moeLayerFreq > 0, "GLM4 MoE Lite MoE frequency must be positive")

        self.layerCount = config.hiddenLayers
        self.firstSparseLayer = config.firstKDenseReplace
        self.moeLayerFrequency = config.moeLayerFreq
        self.routedExpertCount = config.nRoutedExperts
    }

    internal func usesSparseExperts(layerIndex: Int) -> Bool {
        (routedExpertCount ?? 0) > 0
            && layerIndex >= firstSparseLayer
            && layerIndex.isMultiple(of: moeLayerFrequency)
    }
}

internal struct GLM4MoELiteDSAPlan: Equatable, Sendable {
    internal let kinds: [GLMMoEDSAIndexerKind]
    internal let indexHeadDim: Int?
    internal let indexHeads: Int?
    internal let indexTopK: Int?
    internal let queryRank: Int?

    internal init(_ config: GLM4MoELiteConfiguration) {
        self.kinds = config.dsaIndexerKinds
        self.indexHeadDim = config.indexHeadDim
        self.indexHeads = config.indexNHeads
        self.indexTopK = config.indexTopK
        self.queryRank = config.qLoraRank

        if !kinds.isEmpty {
            precondition(config.qLoraRank != nil, "GLM DSA requires q_lora_rank")
            precondition(config.indexHeadDim ?? 0 > 0, "GLM DSA requires index_head_dim")
            precondition(config.indexNHeads ?? 0 > 0, "GLM DSA requires index_n_heads")
            precondition(config.indexTopK ?? 0 > 0, "GLM DSA requires index_topk")
            precondition(
                kinds.count == config.hiddenLayers,
                "GLM DSA schedule must match layer count"
            )
        }
    }

    internal var usesDSA: Bool {
        !kinds.isEmpty
    }

    internal func kind(for layerIndex: Int) -> GLMMoEDSAIndexerKind? {
        guard kinds.indices.contains(layerIndex) else {
            return nil
        }
        return kinds[layerIndex]
    }
}

internal enum GLM4MoELiteScoreFunction: String, Equatable, Sendable {
    case sigmoid
    case softmax

    internal init(_ rawValue: String) {
        self = GLM4MoELiteScoreFunction(rawValue: rawValue) ?? .sigmoid
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

internal struct GLM4MoELiteRoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizesSelectedProbabilities: Bool
    internal let routedScalingFactor: Float
    internal let scoreFunction: GLM4MoELiteScoreFunction

    internal init(_ config: GLM4MoELiteConfiguration) {
        let expertCount = config.nRoutedExperts ?? 0
        let keptGroupCount = min(config.topkGroup, config.nGroup)

        precondition(expertCount > 0, "GLM4 MoE Lite routed expert count must be positive")
        precondition(config.topkMethod == "noaux_tc", "GLM4 MoE Lite only supports noaux_tc routing")
        precondition(config.numExpertsPerTok > 0, "GLM4 MoE Lite top-k must be positive")
        precondition(
            config.numExpertsPerTok <= expertCount,
            "GLM4 MoE Lite top-k cannot exceed routed expert count"
        )
        precondition(config.nGroup > 0, "GLM4 MoE Lite group count must be positive")
        precondition(
            expertCount.isMultiple(of: config.nGroup),
            "GLM4 MoE Lite experts must divide evenly into groups"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= config.nGroup,
            "GLM4 MoE Lite kept-group count must be within group count"
        )

        self.expertCount = expertCount
        self.selectedExpertCount = config.numExpertsPerTok
        self.groupCount = config.nGroup
        self.keptGroupCount = keptGroupCount
        self.normalizesSelectedProbabilities = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.scoreFunction = GLM4MoELiteScoreFunction(config.scoringFunc)
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

private enum GLM4MoELiteExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal typealias GLM4MoELiteKVProjectionSplitPlan = MLAKVProjectionSplitPlan

extension MLAKVProjectionSplitPlan {
    internal init(_ config: GLM4MoELiteConfiguration) {
        self.init(
            headCount: config.attentionHeads,
            keyHeadDimensions: config.qkNopeHeadDim,
            valueHeadDimensions: config.vHeadDim,
            latentDimensions: config.kvLoraRank
        )
    }
}

internal struct GLM4MoELiteSanitizerPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int
    internal let tieWordEmbeddings: Bool
    internal let nextTokenPredictorLayerCount: Int
    internal let kvProjection: GLM4MoELiteKVProjectionSplitPlan

    internal init(_ config: GLM4MoELiteConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.nRoutedExperts ?? 0
        self.tieWordEmbeddings = config.tieWordEmbeddings
        self.nextTokenPredictorLayerCount = config.numNextnPredictLayers
        self.kvProjection = GLM4MoELiteKVProjectionSplitPlan(config)
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        if tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex)"
            packExperts(prefix: "\(prefix).mlp", into: &sanitized)
            splitKVProjection(prefix: "\(prefix).self_attn", into: &sanitized)
        }

        return sanitized.filter { key, _ in
            !key.contains("rotary_emb.inv_freq") && !isExtraPredictorLayer(key)
        }
    }

    private func packExperts(prefix: String, into weights: inout [String: MLXArray]) {
        guard expertCount > 0 else { return }

        for projection in GLM4MoELiteExpertProjection.allCases {
            for tensorName in ["weight", "scales", "biases"] {
                let sourceKeys = (0 ..< expertCount).map { expertIndex in
                    "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                }
                guard sourceKeys.allSatisfy({ weights[$0] != nil }) else {
                    continue
                }
                let tensors = sourceKeys.compactMap { weights.removeValue(forKey: $0) }
                weights["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                    stacked(tensors)
            }
        }
    }

    private func splitKVProjection(prefix: String, into weights: inout [String: MLXArray]) {
        let weightKey = "\(prefix).kv_b_proj.weight"
        guard var projection = weights.removeValue(forKey: weightKey) else {
            return
        }

        let scalesKey = "\(prefix).kv_b_proj.scales"
        let biasesKey = "\(prefix).kv_b_proj.biases"
        let isQuantized = weights[scalesKey] != nil
        var inferredBits = 0
        var inferredGroupSize = 0

        if isQuantized {
            guard let scales = weights.removeValue(forKey: scalesKey),
                let biases = weights.removeValue(forKey: biasesKey)
            else {
                weights[weightKey] = projection
                return
            }
            inferredBits = (projection.dim(-1) * 32) / kvProjection.latentDimensions
            inferredGroupSize = kvProjection.latentDimensions / scales.dim(-1)
            projection = dequantized(
                projection,
                scales: scales,
                biases: biases,
                groupSize: inferredGroupSize,
                bits: inferredBits
            )
        }

        var split = kvProjection.split(weight: projection)
        if isQuantized {
            let (embedQ, embedQScales, embedQBiases) = MLX.quantized(
                split.embedQ,
                groupSize: inferredGroupSize,
                bits: inferredBits
            )
            let (unembedOut, unembedOutScales, unembedOutBiases) = MLX.quantized(
                split.unembedOut,
                groupSize: inferredGroupSize,
                bits: inferredBits
            )
            weights["\(prefix).embed_q.scales"] = embedQScales
            weights["\(prefix).embed_q.biases"] = embedQBiases
            weights["\(prefix).unembed_out.scales"] = unembedOutScales
            weights["\(prefix).unembed_out.biases"] = unembedOutBiases
            split = (embedQ, unembedOut)
        }

        weights["\(prefix).embed_q.weight"] = split.embedQ
        weights["\(prefix).unembed_out.weight"] = split.unembedOut
    }

    private func isExtraPredictorLayer(_ key: String) -> Bool {
        guard nextTokenPredictorLayerCount > 0, key.hasPrefix("model.layers.") else {
            return false
        }
        let rest = key.dropFirst("model.layers.".count)
        guard let end = rest.firstIndex(of: "."), let layerIndex = Int(rest[..<end]) else {
            return false
        }
        return layerIndex >= layerCount
    }
}

// MARK: - Multi-head projections

internal typealias GLM4MoELiteHeadProjection = HeadProjection
internal typealias GLM4MoELiteHeadLinear = HeadLinear
internal typealias GLM4MoELiteQuantizedHeadLinear = QuantizedHeadLinear

// MARK: - GLM DSA Indexer Schedule

enum GLMMoEDSAIndexerKind: String, Codable, Sendable {
    case full
    case shared

    init(token: String) throws {
        switch token.lowercased() {
        case "f", "full":
            self = .full
        case "s", "shared":
            self = .shared
        default:
            throw DecodingError.dataCorrupted(.init(
                codingPath: [],
                debugDescription: "Unsupported GLM DSA indexer type '\(token)'"
            ))
        }
    }
}

enum GLMMoEDSAIndexTopKPattern: Equatable, Sendable, Codable {
    case string(String)
    case entries([String])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
            return
        }
        self = .entries(try container.decode([String].self))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .entries(let values):
            try container.encode(values)
        }
    }

    var tokens: [String] {
        switch self {
        case .string(let value):
            value.map(String.init)
        case .entries(let values):
            values
        }
    }
}

struct GLMMoEDSAIndexerSchedule: Equatable, Sendable {
    let kinds: [GLMMoEDSAIndexerKind]

    init(
        layerCount: Int,
        explicitTypes: [String]?,
        pattern: GLMMoEDSAIndexTopKPattern?,
        frequency: Int,
        skipOffset: Int
    ) throws {
        let parsed: [GLMMoEDSAIndexerKind]
        if let explicitTypes {
            parsed = try Self.parse(explicitTypes)
        } else if let pattern {
            parsed = try Self.parse(pattern.tokens)
        } else {
            parsed = Self.derived(layerCount: layerCount, frequency: frequency, skipOffset: skipOffset)
        }
        try Self.validate(parsed, layerCount: layerCount)
        self.kinds = parsed
    }

    private static func parse(_ values: [String]) throws -> [GLMMoEDSAIndexerKind] {
        try values.map { try GLMMoEDSAIndexerKind(token: $0) }
    }

    private static func derived(
        layerCount: Int,
        frequency: Int,
        skipOffset: Int
    ) -> [GLMMoEDSAIndexerKind] {
        let safeFrequency = max(frequency, 1)
        return (0 ..< layerCount).map { layerIndex in
            max(layerIndex - skipOffset + 1, 0) % safeFrequency == 0 ? .full : .shared
        }
    }

    private static func validate(
        _ kinds: [GLMMoEDSAIndexerKind],
        layerCount: Int
    ) throws {
        guard kinds.count == layerCount else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: [],
                debugDescription: "`indexer_types` must have one entry per hidden layer"
            ))
        }
        guard kinds.first == .full else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: [],
                debugDescription: "The first GLM DSA layer must be a full indexer layer"
            ))
        }
    }
}

// MARK: - DSA Indexer

internal final class GLM4MoEDSAIndexer: Module {
    let nHeads: Int
    let headDim: Int
    let indexTopK: Int
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    let rope: RoPELayer

    init(_ config: GLM4MoELiteConfiguration) {
        let plan = GLM4MoELiteDSAPlan(config)
        guard let indexNHeads = plan.indexHeads,
            let indexHeadDim = plan.indexHeadDim,
            let indexTopK = plan.indexTopK,
            let qLoraRank = plan.queryRank
        else {
            preconditionFailure("GLM DSA indexer requires index_n_heads, index_head_dim, index_topk and q_lora_rank")
        }

        self.nHeads = indexNHeads
        self.headDim = indexHeadDim
        self.indexTopK = indexTopK
        self.softmaxScale = pow(Float(indexHeadDim), -0.5)

        _wqB.wrappedValue = Linear(qLoraRank, indexNHeads * indexHeadDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, indexHeadDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: indexHeadDim)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, indexNHeads, bias: false)

        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        qr: MLXArray,
        mask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray? {
        let (batchSize, sequenceLength, _) = (x.dim(0), x.dim(1), x.dim(2))
        var q = wqB(qr)
        q = q.reshaped(batchSize, sequenceLength, nHeads, headDim).transposed(0, 2, 1, 3)

        var k = wk(x)
        k = kNorm(k)
        k = k.reshaped(batchSize, 1, sequenceLength, headDim)

        if let cache {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
            let emptyValues = MLXArray.zeros([batchSize, 1, sequenceLength, 0], dtype: k.dtype)
            k = cache.update(keys: k, values: emptyValues).0
        } else {
            q = rope(q, offset: 0)
            k = rope(k, offset: 0)
        }

        guard k.dim(2) > indexTopK else {
            return nil
        }

        var scores = q.matmul(k.swappedAxes(-1, -2))
        scores = maximum(scores, MLXArray(0, dtype: scores.dtype))

        var weights = weightsProj(x)
        weights = weights * (1 / Float(nHeads).squareRoot()) * softmaxScale
        weights = weights.transposed(0, 2, 1)[.ellipsis, .newAxis]
        scores = (scores * weights).sum(axis: 1, keepDims: true)

        if let mask {
            scores = MLX.where(mask, scores, attentionMaskFillValue(dtype: scores.dtype))
        }

        return argPartition(scores, kth: -indexTopK, axis: -1)[.ellipsis, (-indexTopK)...]
    }
}

// MARK: - Attention

internal final class GLM4MoELiteAttention: Module {
    let config: GLM4MoELiteConfiguration
    let layout: GLM4MoELiteAttentionLayout
    let dsaPlan: GLM4MoELiteDSAPlan
    let hiddenSize: Int
    let numHeads: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "indexer") var indexer: GLM4MoEDSAIndexer?
    let dsaIndexerKind: GLMMoEDSAIndexerKind?

    init(_ config: GLM4MoELiteConfiguration, layerIdx: Int) {
        self.config = config
        let layout = GLM4MoELiteAttentionLayout(config)
        let dsaPlan = GLM4MoELiteDSAPlan(config)
        self.layout = layout
        self.dsaPlan = dsaPlan
        self.hiddenSize = config.hiddenSize
        self.numHeads = layout.attentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = layout.qLoraRank
        self.qkRopeHeadDim = layout.qkRopeHeadDim
        self.kvLoraRank = layout.kvLoraRank
        self.vHeadDim = layout.valueHeadDim
        self.qkNopeHeadDim = layout.qkNopeHeadDim
        self.qHeadDim = layout.queryHeadDim
        self.scale = layout.attentionScale

        if let qLoraRank {
            _qAProj.wrappedValue = Linear(hiddenSize, qLoraRank, bias: config.attentionBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
            _qBProj.wrappedValue = Linear(qLoraRank, layout.queryProjectionDimensions, bias: false)
        } else {
            _qProj.wrappedValue = Linear(hiddenSize, layout.queryProjectionDimensions, bias: false)
        }

        _kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            layout.compressedKeyValueDimensions,
            bias: config.attentionBias
        )
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _embedQ.wrappedValue = GLM4MoELiteHeadLinear(
            inputDims: qkNopeHeadDim,
            outputDims: kvLoraRank,
            headCount: numHeads
        )
        _unembedOut.wrappedValue = GLM4MoELiteHeadLinear(
            inputDims: kvLoraRank,
            outputDims: vHeadDim,
            headCount: numHeads
        )
        _oProj.wrappedValue = Linear(
            layout.outputProjectionDimensions,
            hiddenSize,
            bias: config.attentionBias
        )

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: GLM4MoELiteAttentionLayout.ropeScalingForRuntime(config.ropeScaling),
            maxPositionEmbeddings: maxPositionEmbeddings
        )

        self.dsaIndexerKind = dsaPlan.kind(for: layerIdx)
        if dsaIndexerKind == .full {
            _indexer.wrappedValue = GLM4MoEDSAIndexer(config)
        }
    }

    private func projectHead(
        _ module: Module,
        _ x: MLXArray,
        transposedWeight: Bool = true
    ) -> MLXArray {
        guard let projection = module as? GLM4MoELiteHeadProjection else {
            preconditionFailure("Unsupported GLM4 MoE Lite head projection module: \(type(of: module))")
        }
        return projection.project(x, transposedWeight: transposedWeight)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x, mask: mask, cache: cache, previousTopKIndices: nil).output
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        previousTopKIndices: MLXArray?
    ) -> (output: MLXArray, topKIndices: MLXArray?) {
        guard dsaIndexerKind != nil else {
            return (standardAttention(x, mask: mask, cache: cache), nil)
        }
        return dsaAttention(x, mask: mask, cache: cache, previousTopKIndices: previousTopKIndices)
    }

    private func standardAttention(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }

        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(compressedKv)

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)

        kvLatent = expandedDimensions(kvLatent, axis: 1)

        qNope = projectHead(embedQ, qNope)

        var keys = concatenated([kvLatent, kPe], axis: -1)
        var values = kvLatent

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let queries = concatenated([qNope, qPe], axis: -1)
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        output = projectHead(unembedOut, output)

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(output)
    }

    private func dsaAttention(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        previousTopKIndices: MLXArray?
    ) -> (output: MLXArray, topKIndices: MLXArray?) {
        let (batchSize, sequenceLength, _) = (x.dim(0), x.dim(1), x.dim(2))
        let cacheList = cache as? CacheList
        let attentionCache = cacheList?[0] ?? cache
        let indexerCache = cacheList.flatMap { $0.layoutCaches.count > 1 ? $0[1] : nil }

        let qr = qALayerNorm!(qAProj!(x))
        var q = qBProj!(qr)
        q = q.reshaped(batchSize, sequenceLength, numHeads, qHeadDim)
            .transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(batchSize, sequenceLength, 1, qkRopeHeadDim)
            .transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(compressedKv)

        qPe = applyRotaryPosition(rope, to: qPe, cache: attentionCache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: attentionCache)
        kvLatent = expandedDimensions(kvLatent, axis: 1)

        if let attentionCache {
            (kvLatent, kPe) = attentionCache.update(keys: kvLatent, values: kPe)
        }

        let inputMask = Self.arrayMask(from: mask)
        let shouldReuseTopK = indexer == nil || isIndexCacheSharedLayer(cacheList)
        var topKIndices: MLXArray?
        if let indexer, !shouldReuseTopK {
            topKIndices = indexer(x, qr: qr, mask: inputMask, cache: indexerCache)
        } else {
            topKIndices = previousTopKIndices
        }

        var activeMask = inputMask
        if let topKIndices {
            if sequenceLength == 1 {
                let gatherIndex = topKIndices[.ellipsis, 0, 0..., .newAxis]
                let latentIndex = broadcast(
                    gatherIndex,
                    to: Array(gatherIndex.shape.dropLast()) + [kvLatent.dim(-1)]
                )
                kvLatent = takeAlong(kvLatent, latentIndex, axis: 2)

                let rotaryIndex = broadcast(
                    gatherIndex,
                    to: Array(gatherIndex.shape.dropLast()) + [kPe.dim(-1)]
                )
                kPe = takeAlong(kPe, rotaryIndex, axis: 2)
                if let inputMask {
                    activeMask = takeAlong(inputMask, topKIndices, axis: -1)
                }
            } else {
                var sparseShape = topKIndices.shape
                sparseShape[sparseShape.count - 1] = kvLatent.dim(2)
                var sparseMask = MLXArray.zeros(sparseShape, dtype: .bool)
                sparseMask = putAlong(sparseMask, topKIndices, values: MLXArray(true), axis: -1)
                if let inputMask {
                    sparseMask = sparseMask & inputMask
                }
                activeMask = sparseMask
            }
        }

        var peScores = (qPe * scale).matmul(kPe.swappedAxes(-1, -2))
        if let activeMask {
            peScores = MLX.where(activeMask, peScores, attentionMaskFillValue(dtype: peScores.dtype))
        }

        let keys: MLXArray
        let values: MLXArray
        if sequenceLength == 1 {
            qNope = projectHead(embedQ, qNope)
            keys = kvLatent
            values = kvLatent
        } else {
            keys = projectHead(embedQ, kvLatent, transposedWeight: false)
            values = projectHead(unembedOut, kvLatent)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qNope,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(peScores)
        )
        if sequenceLength == 1 {
            output = projectHead(unembedOut, output)
        }

        output = output.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, -1)
        return (oProj(output), topKIndices)
    }

    private static func arrayMask(from mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray? {
        switch mask {
        case .array(let array):
            return array
        case .arrays(let arrays):
            return arrays.first
        case .causal, .none:
            return nil
        }
    }

    private func isIndexCacheSharedLayer(_ cacheList: CacheList?) -> Bool {
        config.modelType == "deepseek_v32" && cacheList?.layoutCaches.count == 1
    }
}

internal final class GLM4MoELiteMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM4MoELiteConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize

        _gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

internal final class GLM4MoELiteGate: Module {
    let routingPlan: GLM4MoELiteRoutingPlan

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM4MoELiteConfiguration) {
        let routingPlan = GLM4MoELiteRoutingPlan(config)
        self.routingPlan = routingPlan
        _weight.wrappedValue = zeros([routingPlan.expertCount, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([routingPlan.expertCount])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let routed = routingPlan.route(
            logits: x.matmul(weight.T),
            correctionBias: eScoreCorrectionBias,
            outputDType: x.dtype
        )
        return (routed.indices, routed.scores)
    }
}

internal final class GLM4MoELiteMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: GLM4MoELiteGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoELiteMLP?

    init(_ config: GLM4MoELiteConfiguration) {
        let routingPlan = GLM4MoELiteRoutingPlan(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: routingPlan.expertCount
        )
        _gate.wrappedValue = GLM4MoELiteGate(config)

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = GLM4MoELiteMLP(
                config, intermediateSize: intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

internal final class GLM4MoELiteDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoELiteAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(
        _ config: GLM4MoELiteConfiguration,
        layerIdx: Int,
        layerPlan: GLM4MoELiteLayerPlan
    ) {
        _attention.wrappedValue = GLM4MoELiteAttention(config, layerIdx: layerIdx)

        if layerPlan.usesSparseExperts(layerIndex: layerIdx) {
            _feedForward.wrappedValue = GLM4MoELiteMoE(config)
        } else {
            _feedForward.wrappedValue = GLM4MoELiteMLP(config)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x, mask: mask, cache: cache, previousTopKIndices: nil).output
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        previousTopKIndices: MLXArray?
    ) -> (output: MLXArray, topKIndices: MLXArray?) {
        let attentionResult = attention(
            inputLayerNorm(x),
            mask: mask,
            cache: cache,
            previousTopKIndices: previousTopKIndices
        )
        let r = attentionResult.output
        let h = x + r
        let r2 = feedForward(postAttentionLayerNorm(h))
        return (h + r2, attentionResult.topKIndices)
    }
}

internal final class GLM4MoELiteBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [GLM4MoELiteDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: GLM4MoELiteConfiguration

    init(_ config: GLM4MoELiteConfiguration) {
        precondition(config.vocabularySize > 0, "GLM4 MoE Lite vocabulary size must be positive")
        self.config = config
        let layerPlan = GLM4MoELiteLayerPlan(config)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        _layers.wrappedValue = (0 ..< config.hiddenLayers)
            .map { idx in
                GLM4MoELiteDecoderLayer(config, layerIdx: idx, layerPlan: layerPlan)
            }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(
            h: h,
            cache: Self.attentionMaskCache(from: cache),
            returnArray: config.usesDSA
        )

        var previousTopKIndices: MLXArray?
        for (i, layer) in layers.enumerated() {
            let result = layer(
                h,
                mask: mask,
                cache: cache?[i],
                previousTopKIndices: previousTopKIndices
            )
            h = result.output
            previousTopKIndices = result.topKIndices
        }

        return norm(h)
    }

    private static func attentionMaskCache(from cache: [KVCache]?) -> [KVCache]? {
        guard let first = cache?.first else {
            return cache
        }
        if let cacheList = first as? CacheList {
            return [cacheList[0]]
        }
        return cache
    }
}

internal class GLM4MoELiteModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") internal var model: GLM4MoELiteBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: GLM4MoELiteConfiguration

    internal init(_ args: GLM4MoELiteConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        _model.wrappedValue = GLM4MoELiteBackbone(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
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
        guard configuration.usesDSA else {
            return defaultCache(parameters: parameters)
        }
        return configuration.dsaIndexerKinds.map { kind in
            switch kind {
            case .full:
                return CacheList(
                    Self.makeBaseCache(parameters: parameters),
                    Self.makeBaseCache(parameters: parameters)
                )
            case .shared:
                return CacheList(Self.makeBaseCache(parameters: parameters))
            }
        }
    }

    private func defaultCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { _ in
            Self.makeBaseCache(parameters: parameters)
        }
    }

    private static func makeBaseCache(parameters: GenerateParameters?) -> KVCache {
        if let maxKVSize = parameters?.maxKVSize {
            return RotatingKVCache(
                maxSize: maxKVSize,
                keep: GenerationConstants.rotatingCacheKeepTokens
            )
        }
        return KVCacheSimple()
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        GLM4MoELiteSanitizerPlan(configuration).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}

internal struct GLM4MoELiteConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var topkMethod: String
    var scoringFunc: String
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var numExpertsPerTok: Int
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var ropeParameters: [String: StringOrNumber]?
    var ropeTraditional: Bool
    var attentionBias: Bool
    var attentionDropout: Float
    var partialRotaryFactor: Float
    var tieWordEmbeddings: Bool
    var numNextnPredictLayers: Int
    var indexHeadDim: Int?
    var indexNHeads: Int?
    var indexTopK: Int?
    var indexerTypes: [String]?
    var indexTopKPattern: GLMMoEDSAIndexTopKPattern?
    var indexTopKFreq: Int
    var indexSkipTopKOffset: Int
    var dsaIndexerKinds: [GLMMoEDSAIndexerKind]

    var usesDSA: Bool {
        !dsaIndexerKinds.isEmpty
    }

    func dsaIndexerKind(for layerIndex: Int) -> GLMMoEDSAIndexerKind? {
        guard dsaIndexerKinds.indices.contains(layerIndex) else {
            return nil
        }
        return dsaIndexerKinds[layerIndex]
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
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case ropeTraditional = "rope_traditional"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopK = "index_topk"
        case indexerTypes = "indexer_types"
        case indexTopKPattern = "index_topk_pattern"
        case indexTopKFreq = "index_topk_freq"
        case indexSkipTopKOffset = "index_skip_topk_offset"
    }

    internal init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<GLM4MoELiteConfiguration.CodingKeys> =
            try decoder.container(keyedBy: GLM4MoELiteConfiguration.CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        self.kvLoraRank = try container.decode(Int.self, forKey: .kvLoraRank)
        self.qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank)
        self.qkRopeHeadDim = try container.decode(Int.self, forKey: .qkRopeHeadDim)
        self.qkNopeHeadDim = try container.decode(Int.self, forKey: .qkNopeHeadDim)
        self.vHeadDim = try container.decode(Int.self, forKey: .vHeadDim)
        self.topkMethod =
            try container.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
        self.scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.nGroup = try container.decode(Int.self, forKey: .nGroup)
        self.topkGroup = try container.decode(Int.self, forKey: .topkGroup)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.moeLayerFreq = try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        self.firstKDenseReplace = try container.decode(Int.self, forKey: .firstKDenseReplace)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeParameters
        )
        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        } else if let ropeTheta = ropeParameters?["rope_theta"]?.asFloat() {
            self.ropeTheta = ropeTheta
        } else {
            self.ropeTheta = 10_000
        }
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling) ?? ropeParameters
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.attentionDropout =
            try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? false
        self.numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
        self.indexHeadDim = try container.decodeIfPresent(Int.self, forKey: .indexHeadDim)
        self.indexNHeads = try container.decodeIfPresent(Int.self, forKey: .indexNHeads)
        self.indexTopK = try container.decodeIfPresent(Int.self, forKey: .indexTopK)
        self.indexerTypes = try container.decodeIfPresent([String].self, forKey: .indexerTypes)
        self.indexTopKPattern = try container.decodeIfPresent(
            GLMMoEDSAIndexTopKPattern.self,
            forKey: .indexTopKPattern
        )
        self.indexTopKFreq = try container.decodeIfPresent(Int.self, forKey: .indexTopKFreq) ?? 1
        self.indexSkipTopKOffset =
            try container.decodeIfPresent(Int.self, forKey: .indexSkipTopKOffset) ?? 2

        if modelType == "glm_moe_dsa" || indexTopK != nil || indexerTypes != nil || indexTopKPattern != nil {
            guard let indexTopK, indexTopK > 0,
                let indexNHeads, indexNHeads > 0,
                let indexHeadDim, indexHeadDim > 0
            else {
                throw DecodingError.dataCorrupted(.init(
                    codingPath: [CodingKeys.indexTopK],
                    debugDescription: "GLM DSA requires positive index_topk, index_n_heads and index_head_dim"
                ))
            }
            let schedule = try GLMMoEDSAIndexerSchedule(
                layerCount: hiddenLayers,
                explicitTypes: indexerTypes,
                pattern: indexTopKPattern,
                frequency: indexTopKFreq,
                skipOffset: indexSkipTopKOffset
            )
            self.dsaIndexerKinds = schedule.kinds
        } else {
            self.dsaIndexerKinds = []
        }
    }
}

// MARK: - LoRA

extension GLM4MoELiteModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        let targets = configuration.qLoraRank == nil
            ? ["q_proj", "kv_a_proj_with_mqa"]
            : ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"]
        return model.layers.map { ($0.attention, targets) }
    }
}
