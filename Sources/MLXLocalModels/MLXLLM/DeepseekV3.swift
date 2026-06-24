import Darwin
import MLX
import MLXFast
import MLXNN

internal struct DeepseekV3Configuration: Codable, Equatable, Sendable {
    internal var vocabSize: Int
    internal var hiddenSize: Int
    internal var intermediateSize: Int
    internal var moeIntermediateSize: Int
    internal var numHiddenLayers: Int
    internal var numAttentionHeads: Int
    internal var numKeyValueHeads: Int
    internal var nSharedExperts: Int?
    internal var nRoutedExperts: Int?
    internal var routedScalingFactor: Float
    internal var kvLoraRank: Int
    internal var qLoraRank: Int?
    internal var qkRopeHeadDim: Int
    internal var vHeadDim: Int
    internal var qkNopeHeadDim: Int
    internal var normTopkProb: Bool
    internal var nGroup: Int?
    internal var topkGroup: Int?
    internal var numExpertsPerTok: Int?
    internal var moeLayerFreq: Int
    internal var firstKDenseReplace: Int
    internal var maxPositionEmbeddings: Int
    internal var rmsNormEps: Float
    internal var ropeTheta: Float
    internal var ropeScaling: [String: StringOrNumber]?
    internal var attentionBias: Bool

    internal init(
        vocabSize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int? = nil,
        nSharedExperts: Int? = nil,
        nRoutedExperts: Int? = nil,
        routedScalingFactor: Float = 1,
        kvLoraRank: Int,
        qLoraRank: Int? = nil,
        qkRopeHeadDim: Int,
        vHeadDim: Int,
        qkNopeHeadDim: Int,
        normTopkProb: Bool = false,
        nGroup: Int? = nil,
        topkGroup: Int? = nil,
        numExpertsPerTok: Int? = nil,
        moeLayerFreq: Int = 1,
        firstKDenseReplace: Int = Int.max,
        maxPositionEmbeddings: Int = 4_096,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        attentionBias: Bool = false
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads ?? numAttentionHeads
        self.nSharedExperts = nSharedExperts
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = routedScalingFactor
        self.kvLoraRank = kvLoraRank
        self.qLoraRank = qLoraRank.flatMap { $0 > 0 ? $0 : nil }
        self.qkRopeHeadDim = qkRopeHeadDim
        self.vHeadDim = vHeadDim
        self.qkNopeHeadDim = qkNopeHeadDim
        self.normTopkProb = normTopkProb
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.numExpertsPerTok = numExpertsPerTok
        self.moeLayerFreq = max(1, moeLayerFreq)
        self.firstKDenseReplace = firstKDenseReplace
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        let moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? intermediateSize
        let numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        let numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        let kvLoraRank = try container.decode(Int.self, forKey: .kvLoraRank)
        let qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank)
        let qkRopeHeadDim = try container.decode(Int.self, forKey: .qkRopeHeadDim)
        let vHeadDim = try container.decode(Int.self, forKey: .vHeadDim)
        let qkNopeHeadDim = try container.decode(Int.self, forKey: .qkNopeHeadDim)

        self.init(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            moeIntermediateSize: moeIntermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads),
            nSharedExperts: try container.decodeIfPresent(Int.self, forKey: .nSharedExperts),
            nRoutedExperts: try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts),
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 1,
            kvLoraRank: kvLoraRank,
            qLoraRank: qLoraRank,
            qkRopeHeadDim: qkRopeHeadDim,
            vHeadDim: vHeadDim,
            qkNopeHeadDim: qkNopeHeadDim,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false,
            nGroup: try container.decodeIfPresent(Int.self, forKey: .nGroup),
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup),
            numExpertsPerTok: try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok),
            moeLayerFreq: try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1,
            firstKDenseReplace: try container.decodeIfPresent(
                Int.self,
                forKey: .firstKDenseReplace
            ) ?? Int.max,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 4_096,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-6,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false
        )
    }

    internal enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
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
        case attentionBias = "attention_bias"
    }
}

internal struct DeepseekV3AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let kvLoraRank: Int
    internal let queryLowRank: Int?
    internal let nopeHeadSize: Int
    internal let ropeHeadSize: Int
    internal let valueHeadSize: Int
    internal let queryHeadSize: Int
    internal let queryProjectionSize: Int
    internal let compressedKeyValueSize: Int
    internal let keyValueProjectionSize: Int
    internal let outputProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: DeepseekV3Configuration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.numAttentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvLoraRank > 0, "kv_lora_rank must be positive")
        precondition(config.qkRopeHeadDim > 0, "qk_rope_head_dim must be positive")
        precondition(config.vHeadDim > 0, "v_head_dim must be positive")
        precondition(config.qkNopeHeadDim > 0, "qk_nope_head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.numAttentionHeads
        self.kvLoraRank = config.kvLoraRank
        self.queryLowRank = config.qLoraRank
        self.nopeHeadSize = config.qkNopeHeadDim
        self.ropeHeadSize = config.qkRopeHeadDim
        self.valueHeadSize = config.vHeadDim
        self.queryHeadSize = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.queryProjectionSize = config.numAttentionHeads * queryHeadSize
        self.compressedKeyValueSize = config.kvLoraRank + config.qkRopeHeadDim
        self.keyValueProjectionSize = config.numAttentionHeads * (
            config.qkNopeHeadDim + config.vHeadDim
        )
        self.outputProjectionSize = config.numAttentionHeads * config.vHeadDim
        self.attentionScale = powf(Float(queryHeadSize), -0.5)
    }

    internal func scaledAttentionScale(for ropePlan: DeepseekV3YarnPlan) -> Float {
        attentionScale * ropePlan.attentionScaleMultiplier
    }
}

internal struct DeepseekV3YarnPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let maxPositionEmbeddings: Int
    internal let base: Float
    internal let scalingFactor: Float
    internal let originalMaxPositionEmbeddings: Int
    internal let betaFast: Float
    internal let betaSlow: Float
    internal let magnitudeScale: Float
    internal let magnitudeScaleAllDimensions: Float

    internal init(_ config: DeepseekV3Configuration, dimensions: Int) {
        precondition(dimensions > 0 && dimensions % 2 == 0, "YaRN dimensions must be even")

        let scaling = config.ropeScaling
        self.dimensions = dimensions
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.base = config.ropeTheta
        self.scalingFactor = scaling?["factor"]?.asFloat() ?? 1
        self.originalMaxPositionEmbeddings = scaling?[
            "original_max_position_embeddings"
        ]?.asInt() ?? 4_096
        self.betaFast = scaling?["beta_fast"]?.asFloat() ?? 32
        self.betaSlow = scaling?["beta_slow"]?.asFloat() ?? 1
        self.magnitudeScale = scaling?["mscale"]?.asFloat() ?? 1
        self.magnitudeScaleAllDimensions = scaling?["mscale_all_dim"]?.asFloat() ?? 0
    }

    internal var rotaryInputScale: Float {
        Self.magnitude(scale: scalingFactor, multiplier: magnitudeScale)
            / Self.magnitude(scale: scalingFactor, multiplier: magnitudeScaleAllDimensions)
    }

    internal var attentionScaleMultiplier: Float {
        guard magnitudeScaleAllDimensions != 0 else {
            return 1
        }
        let scale = Self.magnitude(
            scale: scalingFactor,
            multiplier: magnitudeScaleAllDimensions
        )
        return scale * scale
    }

    internal func correctionRange() -> (low: Int, high: Int) {
        let low = Int(floorf(Self.correctionDimension(
            rotations: betaFast,
            dimensions: Float(dimensions),
            base: base,
            originalMaxPositionEmbeddings: Float(originalMaxPositionEmbeddings)
        )))
        let high = Int(ceilf(Self.correctionDimension(
            rotations: betaSlow,
            dimensions: Float(dimensions),
            base: base,
            originalMaxPositionEmbeddings: Float(originalMaxPositionEmbeddings)
        )))
        return (max(low, 0), min(high, dimensions - 1))
    }

    internal static func correctionDimension(
        rotations: Float,
        dimensions: Float,
        base: Float,
        originalMaxPositionEmbeddings: Float
    ) -> Float {
        dimensions * logf(originalMaxPositionEmbeddings / (rotations * 2 * Float.pi))
            / (2 * logf(base))
    }

    internal static func magnitude(scale: Float, multiplier: Float) -> Float {
        scale <= 1 ? 1 : 0.1 * multiplier * logf(scale) + 1
    }
}

internal struct DeepseekV3RoutingPlan: Equatable, Sendable {
    internal let routedExperts: Int
    internal let expertsPerToken: Int
    internal let groupCount: Int
    internal let keptGroupCount: Int
    internal let normalizeTopK: Bool
    internal let routedScalingFactor: Float

    internal init(_ config: DeepseekV3Configuration) {
        let routedExperts = config.nRoutedExperts ?? 1
        let expertsPerToken = config.numExpertsPerTok ?? 1
        let groupCount = config.nGroup ?? 1
        let keptGroupCount = config.topkGroup ?? groupCount

        precondition(routedExperts > 0, "n_routed_experts must be positive")
        precondition(expertsPerToken > 0, "num_experts_per_tok must be positive")
        precondition(groupCount > 0, "n_group must be positive")
        precondition(
            routedExperts % groupCount == 0,
            "n_routed_experts must be divisible by n_group"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= groupCount,
            "topk_group must be within n_group"
        )
        precondition(
            expertsPerToken <= routedExperts,
            "num_experts_per_tok cannot exceed n_routed_experts"
        )

        self.routedExperts = routedExperts
        self.expertsPerToken = expertsPerToken
        self.groupCount = groupCount
        self.keptGroupCount = keptGroupCount
        self.normalizeTopK = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
    }

    internal var expertsPerGroup: Int {
        routedExperts / groupCount
    }

    internal var droppedGroupCount: Int {
        groupCount - keptGroupCount
    }
}

private final class DeepseekV3YarnRotaryEmbedding {
    private let plan: DeepseekV3YarnPlan
    private let frequencies: MLXArray

    init(_ plan: DeepseekV3YarnPlan) {
        self.plan = plan
        self.frequencies = Self.frequencies(for: plan)
    }

    func callAsFunction(_ input: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            plan.rotaryInputScale != 1 ? input * plan.rotaryInputScale : input,
            dimensions: plan.dimensions,
            traditional: true,
            base: nil,
            scale: 1,
            offset: offset,
            freqs: frequencies
        )
    }

    private static func frequencies(for plan: DeepseekV3YarnPlan) -> MLXArray {
        let positions = MLXArray(stride(from: 0, to: plan.dimensions, by: 2)).asType(.float32)
        let baseFrequencies = MLX.pow(plan.base, positions / Float(plan.dimensions))
        let interpolatedFrequencies = plan.scalingFactor * baseFrequencies
        let range = plan.correctionRange()
        let mask = 1 - linearRampMask(
            min: Float(range.low),
            max: Float(range.high),
            count: plan.dimensions / 2
        )

        return (interpolatedFrequencies * baseFrequencies)
            / (interpolatedFrequencies * mask + baseFrequencies * (1 - mask))
    }

    private static func linearRampMask(min minimum: Float, max maximum: Float, count: Int)
        -> MLXArray {
        let adjustedMaximum = minimum == maximum ? maximum + 0.001 : maximum
        let ramp = (MLXArray(0 ..< count).asType(.float32) - minimum) / (
            adjustedMaximum - minimum
        )
        return clip(ramp, min: 0, max: 1)
    }
}

private final class DeepseekV3SelfAttention: Module {
    private let layout: DeepseekV3AttentionLayout
    private let rope: DeepseekV3YarnRotaryEmbedding
    private let attentionScale: Float

    @ModuleInfo(key: "q_proj") private var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") private var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") private var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") private var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") private var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") private var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") private var kvBProj: Linear
    @ModuleInfo(key: "o_proj") private var oProj: Linear

    init(config: DeepseekV3Configuration) {
        self.layout = DeepseekV3AttentionLayout(config)
        let ropePlan = DeepseekV3YarnPlan(config, dimensions: layout.ropeHeadSize)
        self.rope = DeepseekV3YarnRotaryEmbedding(ropePlan)
        self.attentionScale = layout.scaledAttentionScale(for: ropePlan)

        if let rank = layout.queryLowRank {
            self._qAProj.wrappedValue = Linear(
                layout.hiddenSize,
                rank,
                bias: config.attentionBias
            )
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: rank)
            self._qBProj.wrappedValue = Linear(
                rank,
                layout.queryProjectionSize,
                bias: false
            )
        } else {
            self._qProj.wrappedValue = Linear(
                layout.hiddenSize,
                layout.queryProjectionSize,
                bias: false
            )
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            layout.hiddenSize,
            layout.compressedKeyValueSize,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: layout.kvLoraRank)
        self._kvBProj.wrappedValue = Linear(
            layout.kvLoraRank,
            layout.keyValueProjectionSize,
            bias: false
        )
        self._oProj.wrappedValue = Linear(
            layout.outputProjectionSize,
            layout.hiddenSize,
            bias: config.attentionBias
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)

        let projectedQueries: MLXArray
        if let qProj {
            projectedQueries = qProj(input)
        } else {
            projectedQueries = qBProj!(qALayerNorm!(qAProj!(input)))
        }

        let queries = projectedQueries
            .reshaped(batchSize, sequenceLength, layout.attentionHeads, layout.queryHeadSize)
            .transposed(0, 2, 1, 3)
        let queryParts = split(queries, indices: [layout.nopeHeadSize], axis: -1)
        var queryRope = queryParts[1]

        let compressedKeyValues = kvAProjWithMqa(input)
        let compressedParts = split(compressedKeyValues, indices: [layout.kvLoraRank], axis: -1)
        let latentKeyValues = compressedParts[0]
        var keyRope = compressedParts[1]
            .reshaped(batchSize, sequenceLength, 1, layout.ropeHeadSize)
            .transposed(0, 2, 1, 3)

        let projectedKeyValues = kvBProj(kvALayerNorm(latentKeyValues))
            .reshaped(batchSize, sequenceLength, layout.attentionHeads, -1)
            .transposed(0, 2, 1, 3)
        let keyValueParts = split(projectedKeyValues, indices: [layout.nopeHeadSize], axis: -1)

        let offset = cache?.offset ?? 0
        queryRope = rope(queryRope, offset: offset)
        keyRope = rope(keyRope, offset: offset)

        let keys = concatenated(
            [
                keyValueParts[0],
                repeated(keyRope, count: layout.attentionHeads, axis: 1),
            ],
            axis: -1
        )
        let values = keyValueParts[1]
        let finalQueries = concatenated([queryParts[0], queryRope], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: finalQueries,
            keys: keys,
            values: values,
            cache: cache,
            scale: attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return oProj(output)
    }
}

@inline(__always)
private func deepseekV3ClippedSilu(_ input: MLXArray) -> MLXArray {
    clip(input * sigmoid(input), min: -100, max: 100)
}

private final class DeepseekV3DenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProj: Linear
    @ModuleInfo(key: "up_proj") private var upProj: Linear
    @ModuleInfo(key: "down_proj") private var downProj: Linear

    init(config: DeepseekV3Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let hiddenSize = hiddenSize ?? config.hiddenSize
        let intermediateSize = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProj(silu(gateProj(input)) * upProj(input))
    }
}

private final class DeepseekV3MoEGate: Module {
    private let plan: DeepseekV3RoutingPlan

    @ModuleInfo(key: "weight") private var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") private var expertScoreCorrectionBias: MLXArray

    init(config: DeepseekV3Configuration) {
        self.plan = DeepseekV3RoutingPlan(config)
        self._weight.wrappedValue = zeros([plan.routedExperts, config.hiddenSize])
        self._expertScoreCorrectionBias.wrappedValue = zeros([plan.routedExperts])
    }

    func callAsFunction(_ input: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)

        let routedScores = sigmoid(input.matmul(weight.T))
        var selectionScores = routedScores + expertScoreCorrectionBias

        if plan.droppedGroupCount > 0 {
            let groupedScores = selectionScores.reshaped(
                batchSize,
                sequenceLength,
                plan.groupCount,
                plan.expertsPerGroup
            )
            let topTwoCount = min(2, plan.expertsPerGroup)
            let groupScoreIndices = argPartition(
                -groupedScores,
                kth: topTwoCount - 1,
                axis: -1
            )[.ellipsis, ..<topTwoCount]
            let groupScores = takeAlong(groupedScores, groupScoreIndices, axis: -1)
                .sum(axis: -1, keepDims: true)
            let droppedGroupIndices = argPartition(
                groupScores,
                kth: plan.droppedGroupCount - 1,
                axis: -2
            )[.ellipsis, ..<plan.droppedGroupCount, 0...]
            let broadcastedIndices = broadcast(
                droppedGroupIndices,
                to: [
                    batchSize,
                    sequenceLength,
                    plan.droppedGroupCount,
                    plan.expertsPerGroup,
                ]
            )
            selectionScores = putAlong(
                groupedScores,
                broadcastedIndices,
                values: MLXArray(0),
                axis: -2
            )
            selectionScores = flattened(selectionScores, start: -2, end: -1)
        }

        let expertIndices = argPartition(
            -selectionScores,
            kth: plan.expertsPerToken - 1,
            axis: -1
        )[.ellipsis, ..<plan.expertsPerToken]
        var expertScores = takeAlong(routedScores, expertIndices, axis: -1)

        if plan.expertsPerToken > 1, plan.normalizeTopK {
            expertScores = expertScores / (expertScores.sum(axis: -1, keepDims: true) + 1e-20)
            expertScores = expertScores * plan.routedScalingFactor
        }

        return (expertIndices, expertScores)
    }
}

private final class DeepseekV3MoE: Module, UnaryLayer {
    private let plan: DeepseekV3RoutingPlan
    @ModuleInfo(key: "gate") private var gate: DeepseekV3MoEGate
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") private var sharedExperts: DeepseekV3DenseMLP?

    init(config: DeepseekV3Configuration) {
        self.plan = DeepseekV3RoutingPlan(config)
        self._gate.wrappedValue = DeepseekV3MoEGate(config: config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: plan.routedExperts,
            activation: deepseekV3ClippedSilu
        )

        if let sharedExpertCount = config.nSharedExperts, sharedExpertCount > 0 {
            self._sharedExperts.wrappedValue = DeepseekV3DenseMLP(
                config: config,
                intermediateSize: config.moeIntermediateSize * sharedExpertCount
            )
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let routing = gate(input)
        var output = switchMLP(input, routing.indices)
        output = (output * routing.scores[.ellipsis, .newAxis]).sum(axis: -2)

        if let sharedExperts {
            output = output + sharedExperts(input)
        }
        return output
    }
}

private final class DeepseekV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: DeepseekV3SelfAttention
    private let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV3Configuration, layerIndex: Int) {
        self._selfAttention.wrappedValue = DeepseekV3SelfAttention(config: config)

        if config.nRoutedExperts != nil,
           layerIndex >= config.firstKDenseReplace,
           layerIndex % config.moeLayerFreq == 0 {
            self.mlp = DeepseekV3MoE(config: config)
        } else {
            self.mlp = DeepseekV3DenseMLP(config: config)
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
        let attended = selfAttention(inputLayerNorm(input), mask: mask, cache: cache)
        let hidden = input + attended
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}

private final class DeepseekV3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") private var embedTokens: Embedding
    fileprivate let layers: [DeepseekV3DecoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(config: DeepseekV3Configuration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV3DecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embedTokens(input)
        let attentionMask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: attentionMask, cache: cache?[index])
        }

        return norm(hiddenStates)
    }
}

internal final class DeepseekV3Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let kvHeads: [Int]
    internal let vocabularySize: Int

    private let config: DeepseekV3Configuration
    fileprivate let model: DeepseekV3Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    init(_ config: DeepseekV3Configuration) {
        self.config = config
        self.kvHeads = Array(repeating: config.numAttentionHeads, count: config.numHiddenLayers)
        self.vocabularySize = config.vocabSize
        self.model = DeepseekV3Backbone(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenState = lastTokenHiddenState(model(input.tokens, cache: cache))
        return greedyTokenOutput(logits: lmHead(hiddenState), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .block(),
            sidecarPolicy: .dropActivationScale
        ).weights

        for layerIndex in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let expertWeights = (0 ..< (config.nRoutedExperts ?? 0)).compactMap {
                        weights["\(prefix).experts.\($0).\(projection).\(key)"]
                    }
                    guard expertWeights.count == config.nRoutedExperts else {
                        continue
                    }
                    sanitized["\(prefix).switch_mlp.\(projection).\(key)"] = stacked(expertWeights)
                }
            }
        }

        return sanitized.filter { key, _ in
            !key.contains(".rotary_emb.inv_freq")
                && !key.contains(".mlp.experts.")
                && !Self.isExtraPredictorLayer(key, layerCount: config.numHiddenLayers)
        }
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        let targets = config.qLoraRank == nil
            ? ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
            : ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
        return model.layers.map { ($0.selfAttention, targets) }
    }

    private static func isExtraPredictorLayer(_ key: String, layerCount: Int) -> Bool {
        guard key.hasPrefix("model.layers.") else {
            return false
        }

        let remainder = key.dropFirst("model.layers.".count)
        guard let end = remainder.firstIndex(of: "."),
              let layerIndex = Int(remainder[..<end]) else {
            return false
        }
        return layerIndex >= layerCount
    }
}
