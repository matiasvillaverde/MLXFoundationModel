import Foundation
import MLX
import MLXFast
import MLXNN

internal struct LongcatFlashConfiguration: Codable, Equatable, Sendable {
    internal var modelType: String
    internal var attentionMethod: String?
    internal var zeroExpertType: String
    internal var hiddenSize: Int
    internal var feedForwardHiddenSize: Int
    internal var moeTopK: Int
    internal var expertFeedForwardHiddenSize: Int
    internal var routedExperts: Int
    internal var zeroExpertCount: Int
    internal var layerCount: Int
    internal var vocabularySize: Int
    internal var maxPositionEmbeddings: Int
    internal var attentionHeads: Int
    internal var kvLoraRank: Int
    internal var qLoraRank: Int?
    internal var qkRopeHeadDim: Int
    internal var qkNopeHeadDim: Int
    internal var valueHeadDim: Int
    internal var routedScalingFactor: Float
    internal var rmsNormEps: Float
    internal var ropeTheta: Float
    internal var scaleQueryLora: Bool
    internal var scaleKeyValueLora: Bool
    internal var attentionBias: Bool
    internal var normalizeTopK: Bool
    internal var routerBias: Bool
    internal var ropeScaling: [String: StringOrNumber]?
    internal var ngramVocabularySizeRatio: Int
    internal var embeddingNeighborCount: Int
    internal var embeddingSplitCount: Int

    internal init(
        modelType: String = "longcat_flash",
        attentionMethod: String? = nil,
        zeroExpertType: String = "identity",
        hiddenSize: Int,
        feedForwardHiddenSize: Int,
        moeTopK: Int,
        expertFeedForwardHiddenSize: Int,
        routedExperts: Int,
        zeroExpertCount: Int = 0,
        layerCount: Int,
        vocabularySize: Int,
        maxPositionEmbeddings: Int = 4_096,
        attentionHeads: Int,
        kvLoraRank: Int,
        qLoraRank: Int? = nil,
        qkRopeHeadDim: Int,
        qkNopeHeadDim: Int,
        valueHeadDim: Int,
        routedScalingFactor: Float = 1,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10_000,
        scaleQueryLora: Bool = false,
        scaleKeyValueLora: Bool = false,
        attentionBias: Bool = false,
        normalizeTopK: Bool = false,
        routerBias: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        ngramVocabularySizeRatio: Int = 78,
        embeddingNeighborCount: Int = 4,
        embeddingSplitCount: Int = 4
    ) {
        self.modelType = modelType
        self.attentionMethod = attentionMethod
        self.zeroExpertType = zeroExpertType
        self.hiddenSize = hiddenSize
        self.feedForwardHiddenSize = feedForwardHiddenSize
        self.moeTopK = moeTopK
        self.expertFeedForwardHiddenSize = expertFeedForwardHiddenSize
        self.routedExperts = routedExperts
        self.zeroExpertCount = zeroExpertCount
        self.layerCount = layerCount
        self.vocabularySize = vocabularySize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionHeads = attentionHeads
        self.kvLoraRank = kvLoraRank
        self.qLoraRank = qLoraRank.flatMap { $0 > 0 ? $0 : nil }
        self.qkRopeHeadDim = qkRopeHeadDim
        self.qkNopeHeadDim = qkNopeHeadDim
        self.valueHeadDim = valueHeadDim
        self.routedScalingFactor = routedScalingFactor
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.scaleQueryLora = scaleQueryLora
        self.scaleKeyValueLora = scaleKeyValueLora
        self.attentionBias = attentionBias
        self.normalizeTopK = normalizeTopK
        self.routerBias = routerBias
        self.ropeScaling = ropeScaling
        self.ngramVocabularySizeRatio = ngramVocabularySizeRatio
        self.embeddingNeighborCount = embeddingNeighborCount
        self.embeddingSplitCount = embeddingSplitCount
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "longcat_flash",
            attentionMethod: try container.decodeIfPresent(String.self, forKey: .attentionMethod),
            zeroExpertType: try container.decodeIfPresent(String.self, forKey: .zeroExpertType)
                ?? "identity",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            feedForwardHiddenSize: try container.decode(Int.self, forKey: .feedForwardHiddenSize),
            moeTopK: try container.decode(Int.self, forKey: .moeTopK),
            expertFeedForwardHiddenSize: try container.decode(
                Int.self,
                forKey: .expertFeedForwardHiddenSize
            ),
            routedExperts: try container.decode(Int.self, forKey: .routedExperts),
            zeroExpertCount: try container.decodeIfPresent(Int.self, forKey: .zeroExpertCount)
                ?? 0,
            layerCount: try container.decode(Int.self, forKey: .layerCount),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 4_096,
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            kvLoraRank: try container.decode(Int.self, forKey: .kvLoraRank),
            qLoraRank: try container.decodeIfPresent(Int.self, forKey: .qLoraRank),
            qkRopeHeadDim: try container.decode(Int.self, forKey: .qkRopeHeadDim),
            qkNopeHeadDim: try container.decode(Int.self, forKey: .qkNopeHeadDim),
            valueHeadDim: try container.decode(Int.self, forKey: .valueHeadDim),
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 1,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-6,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000,
            scaleQueryLora: try container.decodeIfPresent(Bool.self, forKey: .scaleQueryLora)
                ?? false,
            scaleKeyValueLora: try container.decodeIfPresent(
                Bool.self,
                forKey: .scaleKeyValueLora
            ) ?? false,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            normalizeTopK: try container.decodeIfPresent(Bool.self, forKey: .normalizeTopK)
                ?? false,
            routerBias: try container.decodeIfPresent(Bool.self, forKey: .routerBias)
                ?? false,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            ngramVocabularySizeRatio: try container.decodeIfPresent(
                Int.self,
                forKey: .ngramVocabularySizeRatio
            ) ?? 78,
            embeddingNeighborCount: try container.decodeIfPresent(
                Int.self,
                forKey: .embeddingNeighborCount
            ) ?? 4,
            embeddingSplitCount: try container.decodeIfPresent(
                Int.self,
                forKey: .embeddingSplitCount
            ) ?? 4
        )
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case attentionMethod = "attention_method"
        case zeroExpertType = "zero_expert_type"
        case hiddenSize = "hidden_size"
        case feedForwardHiddenSize = "ffn_hidden_size"
        case moeTopK = "moe_topk"
        case expertFeedForwardHiddenSize = "expert_ffn_hidden_size"
        case routedExperts = "n_routed_experts"
        case zeroExpertCount = "zero_expert_num"
        case layerCount = "num_layers"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionHeads = "num_attention_heads"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case valueHeadDim = "v_head_dim"
        case routedScalingFactor = "routed_scaling_factor"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case scaleQueryLora = "mla_scale_q_lora"
        case scaleKeyValueLora = "mla_scale_kv_lora"
        case attentionBias = "attention_bias"
        case normalizeTopK = "norm_topk_prob"
        case routerBias = "router_bias"
        case ropeScaling = "rope_scaling"
        case ngramVocabularySizeRatio = "ngram_vocab_size_ratio"
        case embeddingNeighborCount = "emb_neighbor_num"
        case embeddingSplitCount = "emb_split_num"
    }
}

private final class LongcatFlashAttention: Module {
    private let headCount: Int
    private let qkRopeHeadDim: Int
    private let qkNopeHeadDim: Int
    private let kvLoraRank: Int
    private let queryHeadDim: Int
    private let attentionScale: Float
    private let queryLoraScale: Float?
    private let keyValueLoraScale: Float?

    @ModuleInfo(key: "q_proj") private var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") private var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") private var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") private var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") private var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") private var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") private var embedQ: Module
    @ModuleInfo(key: "unembed_out") private var unembedOut: Module
    @ModuleInfo(key: "o_proj") private var oProj: Linear

    private let rope: RoPELayer

    init(config: LongcatFlashConfiguration) {
        self.headCount = config.attentionHeads
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.queryHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        var scale = Float(1.0 / Double(queryHeadDim).squareRoot())
        if let ropeScale = config.ropeScaling,
           (ropeScale["mscale_all_dim"]?.asFloat() ?? 0) != 0,
           let factor = ropeScale["factor"]?.asFloat(),
           factor > 1 {
            let multiplier = 0.1 * (ropeScale["mscale_all_dim"]?.asFloat() ?? 0)
                * log(factor) + 1
            scale *= multiplier * multiplier
        }
        self.attentionScale = scale

        if let qLoraRank = config.qLoraRank, config.scaleQueryLora {
            self.queryLoraScale = Float(Double(config.hiddenSize) / Double(qLoraRank)).squareRoot()
        } else {
            self.queryLoraScale = nil
        }
        self.keyValueLoraScale = config.scaleKeyValueLora
            ? Float(Double(config.hiddenSize) / Double(config.kvLoraRank)).squareRoot()
            : nil

        if let qLoraRank = config.qLoraRank {
            self._qAProj.wrappedValue = Linear(
                config.hiddenSize,
                qLoraRank,
                bias: config.attentionBias
            )
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            self._qBProj.wrappedValue = Linear(
                qLoraRank,
                config.attentionHeads * queryHeadDim,
                bias: false
            )
        } else {
            self._qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.attentionHeads * queryHeadDim,
                bias: false
            )
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize,
            config.kvLoraRank + config.qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank)
        self._embedQ.wrappedValue = HeadLinear(
            inputDims: config.qkNopeHeadDim,
            outputDims: config.kvLoraRank,
            headCount: config.attentionHeads
        )
        self._unembedOut.wrappedValue = HeadLinear(
            inputDims: config.kvLoraRank,
            outputDims: config.valueHeadDim,
            headCount: config.attentionHeads
        )
        self._oProj.wrappedValue = Linear(
            config.attentionHeads * config.valueHeadDim,
            config.hiddenSize,
            bias: config.attentionBias
        )

        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)

        let queryProjection: MLXArray
        if let qProj {
            queryProjection = qProj(input)
        } else {
            guard let qAProj, let qALayerNorm, let qBProj else {
                preconditionFailure("LongCat query LoRA projection is incomplete")
            }
            queryProjection = qBProj(qALayerNorm(qAProj(input)))
        }

        var queries = queryProjection
            .reshaped(batchSize, sequenceLength, headCount, queryHeadDim)
            .transposed(0, 2, 1, 3)
        if let queryLoraScale {
            queries = queries * queryLoraScale
        }
        let queryParts = split(queries, indices: [qkNopeHeadDim], axis: -1)
        var queryRope = queryParts[1]

        let compressedKeyValues = kvAProjWithMqa(input)
        let compressedParts = split(compressedKeyValues, indices: [kvLoraRank], axis: -1)
        var latentKeyValues = kvALayerNorm(compressedParts[0])
        if let keyValueLoraScale {
            latentKeyValues = latentKeyValues * keyValueLoraScale
        }
        latentKeyValues = expandedDimensions(latentKeyValues, axis: 1)

        var keyRope = compressedParts[1]
            .reshaped(batchSize, sequenceLength, 1, qkRopeHeadDim)
            .transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queryRope = rope(queryRope, offset: offset)
        keyRope = rope(keyRope, offset: offset)

        if let cache {
            let updated = updateCacheReturningMaterializedKV(
                keys: latentKeyValues,
                values: keyRope,
                cache: cache
            )
            latentKeyValues = updated.keys
            keyRope = updated.values
        }

        let ropeScores = (queryRope * attentionScale).matmul(keyRope.swappedAxes(-1, -2))
        let additiveMask = Self.attentionMask(scores: ropeScores, baseMask: mask)

        let attended: MLXArray
        if sequenceLength == 1 {
            let projectedQueries = projectHead(embedQ, queryParts[0])
            attended = projectHead(
                unembedOut,
                MLXFast.scaledDotProductAttention(
                    queries: projectedQueries,
                    keys: latentKeyValues,
                    values: latentKeyValues,
                    scale: attentionScale,
                    mask: additiveMask
                )
            )
        } else {
            attended = MLXFast.scaledDotProductAttention(
                queries: queryParts[0],
                keys: projectHead(embedQ, latentKeyValues, transposedWeight: false),
                values: projectHead(unembedOut, latentKeyValues),
                scale: attentionScale,
                mask: additiveMask
            )
        }

        return oProj(
            attended
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, sequenceLength, -1)
        )
    }

    private func projectHead(
        _ module: Module,
        _ input: MLXArray,
        transposedWeight: Bool = true
    ) -> MLXArray {
        guard let projection = module as? HeadProjection else {
            preconditionFailure("Unsupported LongCat MLA projection module: \(type(of: module))")
        }
        return projection.project(input, transposedWeight: transposedWeight)
    }

    private static func attentionMask(
        scores: MLXArray,
        baseMask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        switch baseMask {
        case .none:
            return .array(scores)
        case .causal:
            let queryLength = scores.dim(-2)
            let keyLength = scores.dim(-1)
            let queryPositions = MLXArray(0 ..< queryLength) + MLXArray(keyLength - queryLength)
            let keyPositions = MLXArray(0 ..< keyLength)
            let causalMask = greaterEqual(
                expandedDimensions(queryPositions, axis: -1),
                expandedDimensions(keyPositions, axis: -2)
            )
            return .array(MLX.where(
                causalMask,
                scores,
                attentionMaskFillValue(dtype: scores.dtype)
            ))
        case .array(let mask):
            return .array(applying(mask: mask, to: scores))
        case .arrays(let masks):
            return .array(masks.reduce(scores) { current, mask in
                applying(mask: mask, to: current)
            })
        }
    }

    private static func applying(mask: MLXArray, to scores: MLXArray) -> MLXArray {
        if mask.dtype == .bool {
            return MLX.where(mask, scores, attentionMaskFillValue(dtype: scores.dtype))
        }
        return scores + mask
    }
}

private final class LongcatFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProj: Linear
    @ModuleInfo(key: "up_proj") private var upProj: Linear
    @ModuleInfo(key: "down_proj") private var downProj: Linear

    init(config: LongcatFlashConfiguration, expert: Bool = false) {
        let hiddenSize = expert
            ? config.expertFeedForwardHiddenSize
            : config.feedForwardHiddenSize
        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenSize, bias: false)
        self._downProj.wrappedValue = Linear(hiddenSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProj(silu(gateProj(input)) * upProj(input))
    }
}

private final class LongcatFlashRouter: Module {
    private let routedExperts: Int
    private let zeroExpertCount: Int
    private let expertsPerToken: Int
    private let routedScalingFactor: Float
    private let normalizeTopK: Bool

    @ModuleInfo(key: "classifier") private var classifier: Linear
    @ParameterInfo(key: "e_score_correction_bias") private var expertScoreCorrectionBias: MLXArray

    init(config: LongcatFlashConfiguration) {
        self.routedExperts = config.routedExperts
        self.zeroExpertCount = config.zeroExpertCount
        self.expertsPerToken = config.moeTopK
        self.routedScalingFactor = config.routedScalingFactor
        self.normalizeTopK = config.normalizeTopK
        self._classifier.wrappedValue = Linear(
            config.hiddenSize,
            config.routedExperts + config.zeroExpertCount,
            bias: config.routerBias
        )
        self._expertScoreCorrectionBias.wrappedValue = zeros([
            config.routedExperts + config.zeroExpertCount
        ])
    }

    func callAsFunction(_ input: MLXArray) -> (regularIndices: MLXArray, regularWeights: MLXArray,
        identityWeights: MLXArray) {
        let logits = classifier(input)
        let scores = softmax(logits, axis: -1)
        let correctedScores = scores + expertScoreCorrectionBias
        let expertIndices = argPartition(-correctedScores, kth: expertsPerToken - 1, axis: -1)[
            .ellipsis,
            ..<expertsPerToken
        ]
        var expertWeights = takeAlong(scores, expertIndices, axis: -1)

        if normalizeTopK {
            expertWeights = expertWeights / (expertWeights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        expertWeights = expertWeights * routedScalingFactor

        let identityMask = expertIndices .>= routedExperts
        let regularIndices = MLX.where(identityMask, MLXArray(0), expertIndices)
        let regularWeights = MLX.where(identityMask, MLXArray(0.0), expertWeights)
        let identityWeights = zeroExpertCount > 0
            ? MLX.where(identityMask, expertWeights, MLXArray(0.0)).sum(axis: -1, keepDims: true)
            : zeros(input.shape.dropLast() + [1])

        return (regularIndices, regularWeights, identityWeights)
    }
}

private final class LongcatFlashMoE: Module, UnaryLayer {
    private let zeroExpertType: String
    @ModuleInfo(key: "router") private var router: LongcatFlashRouter
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU

    init(config: LongcatFlashConfiguration) {
        self.zeroExpertType = config.zeroExpertType
        self._router.wrappedValue = LongcatFlashRouter(config: config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.expertFeedForwardHiddenSize,
            numExperts: config.routedExperts
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let routing = router(input)
        var output = switchMLP(input, routing.regularIndices)
        output = (output * routing.regularWeights[.ellipsis, .newAxis]).sum(axis: -2)

        if zeroExpertType == "identity" {
            output = output + input * routing.identityWeights
        }
        return output
    }
}

private final class LongcatFlashDecoderLayer: Module {
    @ModuleInfo(key: "mlp") private var shortcutMLP: LongcatFlashMoE
    @ModuleInfo(key: "self_attn") private var selfAttention: [LongcatFlashAttention]
    @ModuleInfo(key: "mlps") private var mlps: [LongcatFlashMLP]
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorms: [RMSNorm]
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorms: [RMSNorm]

    init(config: LongcatFlashConfiguration) {
        self._shortcutMLP.wrappedValue = LongcatFlashMoE(config: config)
        self._selfAttention.wrappedValue = [
            LongcatFlashAttention(config: config),
            LongcatFlashAttention(config: config)
        ]
        self._mlps.wrappedValue = [
            LongcatFlashMLP(config: config),
            LongcatFlashMLP(config: config)
        ]
        self._inputLayerNorms.wrappedValue = [
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        ]
        self._postAttentionLayerNorms.wrappedValue = [
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        ]
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: CacheList?
    ) -> MLXArray {
        var hiddenStates = input
        var shortcutOutput: MLXArray?

        for index in 0 ..< 2 {
            let residual = hiddenStates
            let attentionCache = cache?[index]
            hiddenStates = inputLayerNorms[index](hiddenStates)
            hiddenStates = selfAttention[index](
                hiddenStates,
                mask: mask,
                cache: attentionCache
            )
            hiddenStates = residual + hiddenStates

            let mlpResidual = hiddenStates
            let normalized = postAttentionLayerNorms[index](hiddenStates)
            if index == 0 {
                shortcutOutput = shortcutMLP(normalized)
            }
            hiddenStates = mlps[index](normalized)
            hiddenStates = mlpResidual + hiddenStates

            if index == 1, let shortcutOutput {
                hiddenStates = hiddenStates + shortcutOutput
            }
        }

        return hiddenStates
    }
}

private final class LongcatFlashBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [LongcatFlashDecoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(config: LongcatFlashConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.layerCount).map { _ in
            LongcatFlashDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embedTokens(input)
        let firstAttentionCache = (cache?.first as? CacheList)?[0] ?? cache?.first
        let attentionMask = createAttentionMask(
            h: hiddenStates,
            cache: firstAttentionCache,
            returnArray: true
        )

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                mask: attentionMask,
                cache: cache?[index] as? CacheList
            )
        }

        return norm(hiddenStates)
    }
}

private final class LongcatNgramEmbedding: Module {
    private let embeddingBase: Int
    private let splitCount: Int
    private let neighborCount: Int
    private let vocabularyMods: [[Int]]

    @ModuleInfo(key: "word_embeddings") private var wordEmbeddings: Embedding
    @ModuleInfo(key: "embedders") private var embedders: [Embedding]
    @ModuleInfo(key: "post_projs") private var postProjections: [Linear]

    init(config: LongcatFlashConfiguration) {
        self.embeddingBase = config.ngramVocabularySizeRatio * config.vocabularySize
        self.splitCount = config.embeddingSplitCount
        self.neighborCount = config.embeddingNeighborCount

        let auxiliaryEmbeddings = config.embeddingSplitCount * max(0, config.embeddingNeighborCount - 1)
        let auxiliaryDimensions = max(1, config.hiddenSize / max(1, auxiliaryEmbeddings))
        let embeddingBase = config.ngramVocabularySizeRatio * config.vocabularySize
        self.vocabularyMods = Self.makeVocabularyMods(
            vocabularySize: config.vocabularySize,
            embeddingBase: embeddingBase,
            splitCount: config.embeddingSplitCount,
            neighborCount: config.embeddingNeighborCount
        )

        self._wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._embedders.wrappedValue = (0 ..< auxiliaryEmbeddings).map {
            Embedding(
                embeddingCount: embeddingBase + $0 * 2 + 1,
                dimensions: auxiliaryDimensions
            )
        }
        self._postProjections.wrappedValue = (0 ..< auxiliaryEmbeddings).map { _ in
            Linear(auxiliaryDimensions, config.hiddenSize, bias: false)
        }
    }

    func callAsFunction(_ input: MLXArray, cache: MambaCache?) -> MLXArray {
        let sequenceLength = input.dim(-1)
        let tokenIDs = input.asType(.int64)
        let context: MLXArray
        if let cache {
            if let previousContext = cache[0] {
                context = concatenated([previousContext, tokenIDs], axis: -1)
            } else {
                context = tokenIDs
            }
            let keepStart = max(0, context.dim(-1) - neighborCount + 1)
            cache[0] = context[0..., keepStart...]
        } else {
            context = tokenIDs
        }

        var output = wordEmbeddings(tokenIDs)
        guard neighborCount > 1, splitCount > 0 else {
            return output
        }

        for ngram in 2 ... neighborCount {
            for splitIndex in 0 ..< splitCount {
                let index = (ngram - 2) * splitCount + splitIndex
                let embeddingSize = embeddingBase + index * 2 + 1
                let ids = ngramIDs(
                    context,
                    ngram: ngram,
                    mods: vocabularyMods[index],
                    modulo: embeddingSize
                )[0..., (-sequenceLength)...]
                output = output + postProjections[index](embedders[index](ids))
            }
        }

        return output / Float(1 + splitCount * (neighborCount - 1))
    }

    private func ngramIDs(
        _ input: MLXArray,
        ngram: Int,
        mods: [Int],
        modulo: Int
    ) -> MLXArray {
        var ids = input
        for distance in 1 ..< ngram {
            ids = ids + shiftedRight(input, distance: distance) * mods[distance - 1]
        }
        return ids % modulo
    }

    private func shiftedRight(_ input: MLXArray, distance: Int) -> MLXArray {
        guard distance > 0 else { return input }
        let sequenceLength = input.dim(-1)
        if sequenceLength <= distance {
            return zeros(input.shape, dtype: input.dtype)
        }
        let prefix = zeros([input.dim(0), distance], dtype: input.dtype)
        return concatenated([prefix, input[0..., ..<(sequenceLength - distance)]], axis: -1)
    }

    private static func makeVocabularyMods(
        vocabularySize: Int,
        embeddingBase: Int,
        splitCount: Int,
        neighborCount: Int
    ) -> [[Int]] {
        guard neighborCount > 1, splitCount > 0 else { return [] }
        var result: [[Int]] = []
        for ngram in 2 ... neighborCount {
            for splitIndex in 0 ..< splitCount {
                let index = (ngram - 2) * splitCount + splitIndex
                let modulo = embeddingBase + index * 2 + 1
                var powerModulo = 1
                var mods: [Int] = []
                for _ in 0 ..< (ngram - 1) {
                    powerModulo = (powerModulo * vocabularySize) % modulo
                    mods.append(powerModulo)
                }
                result.append(mods)
            }
        }
        return result
    }
}

private final class LongcatFlashNgramBackbone: Module {
    @ModuleInfo(key: "ngram_embeddings") fileprivate var ngramEmbeddings: LongcatNgramEmbedding
    @ModuleInfo(key: "layers") fileprivate var layers: [LongcatFlashDecoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(config: LongcatFlashConfiguration) {
        self._ngramEmbeddings.wrappedValue = LongcatNgramEmbedding(config: config)
        self._layers.wrappedValue = (0 ..< config.layerCount).map { _ in
            LongcatFlashDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = ngramEmbeddings(input, cache: cache?.first as? MambaCache)
        let attentionCache = cache.map { Array($0.dropFirst()) }
        let firstAttentionCache = (attentionCache?.first as? CacheList)?[0] ?? attentionCache?.first
        let attentionMask = createAttentionMask(
            h: hiddenStates,
            cache: firstAttentionCache,
            returnArray: true
        )

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                mask: attentionMask,
                cache: attentionCache?[index] as? CacheList
            )
        }

        return norm(hiddenStates)
    }
}

internal final class LongcatFlashModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let config: LongcatFlashConfiguration
    private let usesNgramEmbeddings: Bool
    @ModuleInfo(key: "model") private var model: Module
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    init(_ config: LongcatFlashConfiguration) {
        self.config = config
        self.usesNgramEmbeddings = config.modelType == "longcat_flash_ngram"
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.attentionHeads, count: config.layerCount)

        self._model.wrappedValue = usesNgramEmbeddings
            ? LongcatFlashNgramBackbone(config: config)
            : LongcatFlashBackbone(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(hiddenStates(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenState = lastTokenHiddenState(hiddenStates(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: lmHead(hiddenState), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let layerCaches = (0 ..< config.layerCount).map { _ in
            CacheList(KVCacheSimple(), KVCacheSimple()) as KVCache
        }
        if usesNgramEmbeddings {
            return [MambaCache()] + layerCaches
        }
        return layerCaches
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .block(),
            sidecarPolicy: .dropActivationScale
        ).weights

        for layerIndex in 0 ..< config.layerCount {
            packExperts(prefix: "model.layers.\(layerIndex).mlp", into: &sanitized)
            for attentionIndex in 0 ..< 2 {
                splitKVProjection(
                    prefix: "model.layers.\(layerIndex).self_attn.\(attentionIndex)",
                    into: &sanitized
                )
            }
        }

        if usesNgramEmbeddings,
           let embedding = sanitized.removeValue(forKey: "model.embed_tokens.weight") {
            sanitized["model.ngram_embeddings.word_embeddings.weight"] = embedding
        }

        return sanitized.filter { key, _ in
            !key.hasPrefix("model.mtp")
                && !key.contains(".rotary_emb.inv_freq")
        }
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        let targets = config.qLoraRank == nil
            ? ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
            : ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
        let layers = (model as? LongcatFlashNgramBackbone)?.layers
            ?? (model as? LongcatFlashBackbone)?.layers
        return (layers ?? []).flatMap { layer in
            (0 ..< 2).map { index in
                (layer.attentionModules[index], targets)
            }
        }
    }

    private func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        if let ngramBackbone = model as? LongcatFlashNgramBackbone {
            return ngramBackbone(inputs, cache: cache)
        }
        guard let backbone = model as? LongcatFlashBackbone else {
            preconditionFailure("LongCat backbone was not initialized")
        }
        return backbone(inputs, cache: cache)
    }

    private func packExperts(prefix: String, into weights: inout [String: MLXArray]) {
        for (source, destination) in [
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj")
        ] {
            for key in ["weight", "scales", "biases"] {
                let expertKeys = (0 ..< config.routedExperts).compactMap { expertIndex in
                    let destinationKey = "\(prefix).experts.\(expertIndex).\(destination).\(key)"
                    let sourceKey = "\(prefix).experts.\(expertIndex).\(source).\(key)"
                    if weights[destinationKey] != nil {
                        return destinationKey
                    }
                    if weights[sourceKey] != nil {
                        return sourceKey
                    }
                    return nil
                }
                guard expertKeys.count == config.routedExperts else {
                    continue
                }
                let expertWeights = expertKeys.compactMap { weights.removeValue(forKey: $0) }
                weights["\(prefix).switch_mlp.\(destination).\(key)"] = stacked(expertWeights)
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
            guard
                let scales = weights.removeValue(forKey: scalesKey),
                let biases = weights.removeValue(forKey: biasesKey)
            else {
                weights[weightKey] = projection
                return
            }
            inferredBits = (projection.dim(-1) * 32) / config.kvLoraRank
            inferredGroupSize = config.kvLoraRank / scales.dim(-1)
            projection = dequantized(
                projection,
                scales: scales,
                biases: biases,
                groupSize: inferredGroupSize,
                bits: inferredBits
            )
        }

        var split = MLAKVProjectionSplitPlan(
            headCount: config.attentionHeads,
            keyHeadDimensions: config.qkNopeHeadDim,
            valueHeadDimensions: config.valueHeadDim,
            latentDimensions: config.kvLoraRank
        )
        .split(weight: projection)

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
}

extension LongcatFlashDecoderLayer {
    fileprivate var attentionModules: [LongcatFlashAttention] {
        selfAttention
    }
}
