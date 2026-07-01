import Foundation
import MLX
import MLXFast
import MLXNN

internal struct KimiLinearAttentionConfiguration: Codable, Equatable, Sendable {
    var numHeads: Int
    var headDim: Int
    var kdaLayers: [Int]
    var shortConvKernelSize: Int

    init(
        numHeads: Int,
        headDim: Int,
        kdaLayers: [Int],
        shortConvKernelSize: Int = 4
    ) {
        precondition(numHeads > 0, "Kimi linear num_heads must be positive")
        precondition(headDim > 0, "Kimi linear head_dim must be positive")
        precondition(shortConvKernelSize > 1, "Kimi linear short conv kernel must exceed one")
        self.numHeads = numHeads
        self.headDim = headDim
        self.kdaLayers = kdaLayers
        self.shortConvKernelSize = shortConvKernelSize
    }

    private enum CodingKeys: String, CodingKey {
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case kdaLayers = "kda_layers"
        case shortConvKernelSize = "short_conv_kernel_size"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            numHeads: try container.decode(Int.self, forKey: .numHeads),
            headDim: try container.decode(Int.self, forKey: .headDim),
            kdaLayers: try container.decode([Int].self, forKey: .kdaLayers),
            shortConvKernelSize: try container.decodeIfPresent(
                Int.self,
                forKey: .shortConvKernelSize
            ) ?? 4
        )
    }
}

internal struct KimiLinearConfiguration: Codable, Equatable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var intermediateSize: Int
    var headDim: Int
    var ropeTheta: Float
    var rmsNormEps: Float
    var linearAttention: KimiLinearAttentionConfiguration
    var modelMaxLength: Int
    var numExperts: Int
    var moeIntermediateSize: Int
    var kvLoraRank: Int
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var qkNopeHeadDim: Int
    var qkRopeHeadDim: Int
    var valueHeadDim: Int
    var mlaUseNope: Bool
    var numExpertsPerToken: Int
    var numSharedExperts: Int
    var routerActivation: String
    var moeRenormalize: Bool
    var routedScalingFactor: Float
    var firstKDenseReplace: Int
    var moeLayerFrequency: Int
    var useGroupedTopK: Bool
    var expertGroupCount: Int
    var topkGroup: Int

    init(
        modelType: String = "kimi_linear",
        vocabularySize: Int,
        hiddenSize: Int,
        hiddenLayers: Int,
        attentionHeads: Int,
        kvHeads: Int,
        intermediateSize: Int,
        headDim: Int,
        ropeTheta: Float,
        rmsNormEps: Float,
        linearAttention: KimiLinearAttentionConfiguration,
        modelMaxLength: Int,
        numExperts: Int,
        moeIntermediateSize: Int,
        kvLoraRank: Int,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        qkNopeHeadDim: Int? = nil,
        qkRopeHeadDim: Int? = nil,
        valueHeadDim: Int? = nil,
        mlaUseNope: Bool = false,
        numExpertsPerToken: Int = 1,
        numSharedExperts: Int = 0,
        routerActivation: String = "sigmoid",
        moeRenormalize: Bool = true,
        routedScalingFactor: Float = 1,
        firstKDenseReplace: Int = 0,
        moeLayerFrequency: Int = 1,
        useGroupedTopK: Bool = true,
        expertGroupCount: Int = 1,
        topkGroup: Int = 1
    ) {
        precondition(vocabularySize > 0, "Kimi vocabulary size must be positive")
        precondition(hiddenSize > 0, "Kimi hidden size must be positive")
        precondition(hiddenLayers > 0, "Kimi hidden layer count must be positive")
        precondition(attentionHeads > 0, "Kimi attention head count must be positive")
        precondition(kvHeads > 0, "Kimi KV head count must be positive")
        precondition(intermediateSize > 0, "Kimi intermediate size must be positive")
        precondition(headDim > 0, "Kimi head_dim must be positive")
        precondition(kvLoraRank > 0, "Kimi kv_lora_rank must be positive")
        precondition(modelMaxLength > 0, "Kimi model_max_length must be positive")
        precondition(moeLayerFrequency > 0, "Kimi moe_layer_freq must be positive")
        precondition(expertGroupCount > 0, "Kimi num_expert_group must be positive")
        precondition(topkGroup > 0, "Kimi topk_group must be positive")

        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.intermediateSize = intermediateSize
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.linearAttention = linearAttention
        self.modelMaxLength = modelMaxLength
        self.numExperts = numExperts
        self.moeIntermediateSize = moeIntermediateSize
        self.kvLoraRank = kvLoraRank
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.qkNopeHeadDim = qkNopeHeadDim ?? headDim
        self.qkRopeHeadDim = qkRopeHeadDim ?? 0
        self.valueHeadDim = valueHeadDim ?? headDim
        self.mlaUseNope = mlaUseNope
        self.numExpertsPerToken = numExpertsPerToken
        self.numSharedExperts = numSharedExperts
        self.routerActivation = routerActivation
        self.moeRenormalize = moeRenormalize
        self.routedScalingFactor = routedScalingFactor
        self.firstKDenseReplace = firstKDenseReplace
        self.moeLayerFrequency = moeLayerFrequency
        self.useGroupedTopK = useGroupedTopK
        self.expertGroupCount = expertGroupCount
        self.topkGroup = topkGroup
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case linearAttention = "linear_attn_config"
        case modelMaxLength = "model_max_length"
        case numExperts = "num_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case kvLoraRank = "kv_lora_rank"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case valueHeadDim = "v_head_dim"
        case mlaUseNope = "mla_use_nope"
        case numExpertsPerToken = "num_experts_per_token"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case routerActivation = "moe_router_activation_func"
        case moeRenormalize = "moe_renormalize"
        case routedScalingFactor = "routed_scaling_factor"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFrequency = "moe_layer_freq"
        case useGroupedTopK = "use_grouped_topk"
        case expertGroupCount = "num_expert_group"
        case topkGroup = "topk_group"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let numExpertsPerToken = try container.decodeIfPresent(
            Int.self,
            forKey: .numExpertsPerToken
        ) ?? container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "kimi_linear",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            kvHeads: try container.decode(Int.self, forKey: .kvHeads),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            headDim: try container.decode(Int.self, forKey: .headDim),
            ropeTheta: try container.decode(Float.self, forKey: .ropeTheta),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            linearAttention: try container.decode(
                KimiLinearAttentionConfiguration.self,
                forKey: .linearAttention
            ),
            modelMaxLength: try container.decode(Int.self, forKey: .modelMaxLength),
            numExperts: try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0,
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ) ?? container.decode(Int.self, forKey: .intermediateSize),
            kvLoraRank: try container.decode(Int.self, forKey: .kvLoraRank),
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            qkNopeHeadDim: try container.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim),
            qkRopeHeadDim: try container.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim),
            valueHeadDim: try container.decodeIfPresent(Int.self, forKey: .valueHeadDim),
            mlaUseNope: try container.decodeIfPresent(Bool.self, forKey: .mlaUseNope)
                ?? false,
            numExpertsPerToken: numExpertsPerToken ?? 1,
            numSharedExperts: try container.decodeIfPresent(Int.self, forKey: .numSharedExperts)
                ?? 0,
            routerActivation: try container.decodeIfPresent(String.self, forKey: .routerActivation)
                ?? "sigmoid",
            moeRenormalize: try container.decodeIfPresent(Bool.self, forKey: .moeRenormalize)
                ?? true,
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 1,
            firstKDenseReplace: try container.decodeIfPresent(
                Int.self,
                forKey: .firstKDenseReplace
            ) ?? 0,
            moeLayerFrequency: try container.decodeIfPresent(
                Int.self,
                forKey: .moeLayerFrequency
            ) ?? 1,
            useGroupedTopK: try container.decodeIfPresent(Bool.self, forKey: .useGroupedTopK)
                ?? true,
            expertGroupCount: try container.decodeIfPresent(Int.self, forKey: .expertGroupCount)
                ?? 1,
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(linearAttention, forKey: .linearAttention)
        try container.encode(modelMaxLength, forKey: .modelMaxLength)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(kvLoraRank, forKey: .kvLoraRank)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(qkNopeHeadDim, forKey: .qkNopeHeadDim)
        try container.encode(qkRopeHeadDim, forKey: .qkRopeHeadDim)
        try container.encode(valueHeadDim, forKey: .valueHeadDim)
        try container.encode(mlaUseNope, forKey: .mlaUseNope)
        try container.encode(numExpertsPerToken, forKey: .numExpertsPerTok)
        try container.encode(numSharedExperts, forKey: .numSharedExperts)
        try container.encode(routerActivation, forKey: .routerActivation)
        try container.encode(moeRenormalize, forKey: .moeRenormalize)
        try container.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try container.encode(firstKDenseReplace, forKey: .firstKDenseReplace)
        try container.encode(moeLayerFrequency, forKey: .moeLayerFrequency)
        try container.encode(useGroupedTopK, forKey: .useGroupedTopK)
        try container.encode(expertGroupCount, forKey: .expertGroupCount)
        try container.encode(topkGroup, forKey: .topkGroup)
    }
}

internal struct KimiLinearLayerPlan: Equatable, Sendable {
    let isLinear: [Bool]
    let firstLinearLayerIndex: Int?
    let firstAttentionLayerIndex: Int?

    init(_ config: KimiLinearConfiguration) {
        let linearLayers = Set(config.linearAttention.kdaLayers)
        self.isLinear = (0 ..< config.hiddenLayers).map { linearLayers.contains($0 + 1) }
        self.firstLinearLayerIndex = isLinear.firstIndex(of: true)
        self.firstAttentionLayerIndex = isLinear.firstIndex(of: false)
    }

    func isLinearLayer(_ layerIndex: Int) -> Bool {
        isLinear[layerIndex]
    }

    func usesSparseExperts(_ config: KimiLinearConfiguration, layerIndex: Int) -> Bool {
        config.numExperts > 0
            && layerIndex >= config.firstKDenseReplace
            && layerIndex.isMultiple(of: config.moeLayerFrequency)
    }
}

private protocol KimiLinearAttentionLayer {
    var loraTargets: [String] { get }

    func callAsFunction(
        _ input: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

private final class KimiLinearMLAAttention: Module, KimiLinearAttentionLayer {
    private let qkNopeHeadDim: Int
    private let qkRopeHeadDim: Int
    private let queryHeadDim: Int
    private let valueHeadDim: Int
    private let attentionHeads: Int
    private let kvLoraRank: Int
    private let attentionScale: Float

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") private var kvAProjection: Linear
    @ModuleInfo(key: "kv_a_layernorm") private var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") private var embedQ: Module
    @ModuleInfo(key: "unembed_out") private var unembedOut: Module
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: KimiLinearConfiguration) {
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.queryHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.valueHeadDim = config.valueHeadDim
        self.attentionHeads = config.attentionHeads
        self.kvLoraRank = config.kvLoraRank
        self.attentionScale = pow(Float(queryHeadDim), -0.5)

        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * queryHeadDim,
            bias: false
        )
        _kvAProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.kvLoraRank + config.qkRopeHeadDim,
            bias: false
        )
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        _embedQ.wrappedValue = HeadLinear(
            inputDims: qkNopeHeadDim,
            outputDims: kvLoraRank,
            headCount: attentionHeads
        )
        _unembedOut.wrappedValue = HeadLinear(
            inputDims: kvLoraRank,
            outputDims: valueHeadDim,
            headCount: attentionHeads
        )
        _outputProjection.wrappedValue = Linear(
            config.attentionHeads * config.valueHeadDim,
            config.hiddenSize,
            bias: false
        )
    }

    var loraTargets: [String] {
        ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
    }

    func callAsFunction(
        _ input: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)

        let queries = queryProjection(input)
            .reshaped(batchSize, tokenCount, attentionHeads, queryHeadDim)
            .transposed(0, 2, 1, 3)
        let queryParts = split(queries, indices: [qkNopeHeadDim], axis: -1)

        let compressedKeyValues = kvAProjection(input)
        let compressedParts = split(compressedKeyValues, indices: [kvLoraRank], axis: -1)
        var latentKeyValues = expandedDimensions(kvALayerNorm(compressedParts[0]), axis: 1)
        var keyRope = compressedParts[1]
            .reshaped(batchSize, tokenCount, 1, qkRopeHeadDim)
            .transposed(0, 2, 1, 3)

        if let cache {
            let updated = updateCacheReturningMaterializedKV(
                keys: latentKeyValues,
                values: keyRope,
                cache: cache
            )
            latentKeyValues = updated.keys
            keyRope = updated.values
        }

        let positionScores = (queryParts[1] * attentionScale)
            .matmul(keyRope.swappedAxes(-1, -2))
        let additiveMask = Self.attentionMask(scores: positionScores, baseMask: attentionMask)

        let attended: MLXArray
        if tokenCount == 1 {
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

        return outputProjection(
            attended.transposed(0, 2, 1, 3).reshaped(batchSize, tokenCount, -1)
        )
    }

    private func projectHead(
        _ module: Module,
        _ input: MLXArray,
        transposedWeight: Bool = true
    ) -> MLXArray {
        guard let projection = module as? HeadProjection else {
            preconditionFailure("Unsupported Kimi MLA projection module: \(type(of: module))")
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
            return .array(masks.reduce(scores) { currentScores, mask in
                applying(mask: mask, to: currentScores)
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

private final class KimiLinearShortConv1D: Module {
    let channels: Int
    let kernelSize: Int

    @ModuleInfo(key: "conv") private var convolution: Conv1d

    init(channels: Int, kernelSize: Int) {
        self.channels = channels
        self.kernelSize = kernelSize
        _convolution.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: channels,
            bias: false
        )
    }

    func callAsFunction(_ input: MLXArray, state: MLXArray?, mask: MLXArray?) -> (MLXArray, MLXArray) {
        let batchSize = input.dim(0)
        let stateLength = kernelSize - 1
        var tokens = input
        if let mask {
            tokens = MLX.where(
                expandedDimensions(mask, axis: -1),
                tokens,
                MLXArray.zeros(like: tokens)
            )
        }

        let previousState = state ?? MLXArray.zeros(
            [batchSize, stateLength, channels],
            dtype: input.dtype
        )
        let convInput = concatenated([previousState, tokens], axis: 1)
        let nextState = convInput[0..., (1 - kernelSize)..., 0...]
        return (silu(convolution(convInput)), nextState)
    }
}

private final class KimiLinearDeltaAttention: Module, KimiLinearAttentionLayer {
    private let numHeads: Int
    private let headDim: Int
    private let projectionDim: Int
    private let convKernel: Int
    private let scale: Float

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "q_conv") private var queryConv: KimiLinearShortConv1D
    @ModuleInfo(key: "k_conv") private var keyConv: KimiLinearShortConv1D
    @ModuleInfo(key: "v_conv") private var valueConv: KimiLinearShortConv1D
    @ModuleInfo(key: "f_a_proj") private var decayAProjection: Linear
    @ModuleInfo(key: "f_b_proj") private var decayBProjection: Linear
    @ModuleInfo(key: "b_proj") private var betaProjection: Linear
    @ModuleInfo(key: "g_a_proj") private var gateAProjection: Linear
    @ModuleInfo(key: "g_b_proj") private var gateBProjection: Linear
    @ParameterInfo(key: "A_log") private var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") private var dtBias: MLXArray
    @ModuleInfo(key: "o_norm") private var outputNorm: RMSNorm
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: KimiLinearConfiguration) {
        self.numHeads = config.linearAttention.numHeads
        self.headDim = config.linearAttention.headDim
        self.projectionDim = config.linearAttention.numHeads * config.linearAttention.headDim
        self.convKernel = config.linearAttention.shortConvKernelSize
        self.scale = pow(Float(config.linearAttention.headDim), -0.5)

        _queryProjection.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        _queryConv.wrappedValue = KimiLinearShortConv1D(channels: projectionDim, kernelSize: convKernel)
        _keyConv.wrappedValue = KimiLinearShortConv1D(channels: projectionDim, kernelSize: convKernel)
        _valueConv.wrappedValue = KimiLinearShortConv1D(channels: projectionDim, kernelSize: convKernel)
        _decayAProjection.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _decayBProjection.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _betaProjection.wrappedValue = Linear(config.hiddenSize, numHeads, bias: false)
        _gateAProjection.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _gateBProjection.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _aLog.wrappedValue = log(MLXRandom.uniform(low: 1, high: 16, [numHeads]))
        _dtBias.wrappedValue = MLXArray.zeros([projectionDim])
        _outputNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _outputProjection.wrappedValue = Linear(projectionDim, config.hiddenSize, bias: false)
    }

    var loraTargets: [String] {
        [
            "q_proj", "k_proj", "v_proj",
            "f_a_proj", "f_b_proj", "b_proj",
            "g_a_proj", "g_b_proj", "o_proj"
        ]
    }

    func callAsFunction(
        _ input: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)
        let cache = cache as? MambaCache

        let (queryConvOutput, queryState) = queryConv(
            queryProjection(input),
            state: cache?[0],
            mask: ssmMask
        )
        let (keyConvOutput, keyState) = keyConv(
            keyProjection(input),
            state: cache?[1],
            mask: ssmMask
        )
        let (valueConvOutput, valueState) = valueConv(
            valueProjection(input),
            state: cache?[2],
            mask: ssmMask
        )

        if let cache {
            cache[0] = queryState
            cache[1] = keyState
            cache[2] = valueState
        }

        var queries = queryConvOutput.reshaped(batchSize, tokenCount, numHeads, headDim)
        var keys = keyConvOutput.reshaped(batchSize, tokenCount, numHeads, headDim)
        let values = valueConvOutput.reshaped(batchSize, tokenCount, numHeads, headDim)

        queries = MLXArray(scale * scale).asType(input.dtype)
            * MLXFast.rmsNorm(queries, weight: MLXArray.mlxNone, eps: 1e-6)
        keys = MLXArray(scale).asType(input.dtype)
            * MLXFast.rmsNorm(keys, weight: MLXArray.mlxNone, eps: 1e-6)

        let decayInput = decayBProjection(decayAProjection(input))
            .reshaped(batchSize, tokenCount, numHeads, headDim)
        let betaInput = betaProjection(input).reshaped(batchSize, tokenCount, numHeads)

        let decay = computeGatedDeltaG(
            aLog.reshaped(numHeads, 1),
            decayInput,
            dtBias.reshaped(numHeads, headDim)
        )
        let beta = sigmoid(betaInput)
        let (output, nextState) = gatedDeltaOps(
            q: queries,
            k: keys,
            v: values,
            g: decay,
            beta: beta,
            state: cache?[3],
            mask: ssmMask
        )

        if let cache {
            cache[3] = nextState
            cache.offset += tokenCount
        }

        let gate = gateBProjection(gateAProjection(input))
            .reshaped(batchSize, tokenCount, numHeads, headDim)
        let gated = outputNorm(output) * sigmoid(gate)
        return outputProjection(gated.reshaped(batchSize, tokenCount, -1))
    }
}

private final class KimiLinearMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: KimiLinearConfiguration, intermediateSize: Int? = nil) {
        let intermediateSize = intermediateSize ?? config.intermediateSize
        _gateProjection.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _upProjection.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _downProjection.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private struct KimiLinearRoutingPlan: Equatable, Sendable {
    let expertCount: Int
    let selectedExpertCount: Int
    let groupCount: Int
    let keptGroupCount: Int
    let normalizesSelectedScores: Bool
    let routedScalingFactor: Float
    let scoreFunction: BailingMoeScoreFunction
    let useGroupedTopK: Bool

    init(_ config: KimiLinearConfiguration) {
        let keptGroupCount = min(config.topkGroup, config.expertGroupCount)
        precondition(config.numExperts > 0, "Kimi sparse MoE expert count must be positive")
        precondition(config.numExpertsPerToken > 0, "Kimi sparse MoE top-k must be positive")
        precondition(config.numExpertsPerToken <= config.numExperts, "Kimi sparse MoE top-k is too large")
        precondition(
            config.numExperts.isMultiple(of: config.expertGroupCount),
            "Kimi sparse MoE experts must divide groups"
        )
        precondition(
            keptGroupCount > 0 && keptGroupCount <= config.expertGroupCount,
            "Kimi sparse MoE topk_group must fit group count"
        )

        self.expertCount = config.numExperts
        self.selectedExpertCount = config.numExpertsPerToken
        self.groupCount = config.expertGroupCount
        self.keptGroupCount = keptGroupCount
        self.normalizesSelectedScores = config.moeRenormalize
        self.routedScalingFactor = config.routedScalingFactor
        self.scoreFunction = BailingMoeScoreFunction(config.routerActivation)
        self.useGroupedTopK = config.useGroupedTopK
    }

    var expertsPerGroup: Int {
        expertCount / groupCount
    }

    private var droppedGroupCount: Int {
        groupCount - keptGroupCount
    }

    func route(
        logits: MLXArray,
        correctionBias: MLXArray,
        outputDType: DType
    ) -> (scores: MLXArray, indices: MLXArray) {
        let originalScores = scoreFunction.scores(from: logits)
        var selectionScores = originalScores + correctionBias.asType(originalScores.dtype)

        if useGroupedTopK, groupCount > 1, droppedGroupCount > 0 {
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

        if selectedExpertCount > 1, normalizesSelectedScores {
            selectedScores = selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
        }

        return ((selectedScores * routedScalingFactor).asType(outputDType), indices)
    }
}

private final class KimiLinearSparseMoE: Module, UnaryLayer {
    private let plan: KimiLinearRoutingPlan

    @ModuleInfo(key: "gate") private var gate: Linear
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") private var correctionBias: MLXArray
    @ModuleInfo(key: "shared_experts") private var sharedExperts: KimiLinearMLP?

    init(_ config: KimiLinearConfiguration) {
        self.plan = KimiLinearRoutingPlan(config)
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        _correctionBias.wrappedValue = MLXArray.zeros([config.numExperts])
        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = KimiLinearMLP(
                config,
                intermediateSize: config.moeIntermediateSize * config.numSharedExperts
            )
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let routed = plan.route(
            logits: gate(input),
            correctionBias: correctionBias,
            outputDType: input.dtype
        )
        var output = switchMLP(input, routed.indices)
        output = (output * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            output = output + sharedExperts(input)
        }
        return output
    }
}

private final class KimiLinearDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: Module & KimiLinearAttentionLayer
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: KimiLinearConfiguration, layerIndex: Int, layerPlan: KimiLinearLayerPlan) {
        self.isLinear = layerPlan.isLinearLayer(layerIndex)
        if isLinear {
            _selfAttention.wrappedValue = KimiLinearDeltaAttention(config)
        } else {
            _selfAttention.wrappedValue = KimiLinearMLAAttention(config)
        }

        if layerPlan.usesSparseExperts(config, layerIndex: layerIndex) {
            _feedForward.wrappedValue = KimiLinearSparseMoE(config)
        } else {
            _feedForward.wrappedValue = KimiLinearMLP(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let hidden = input + selfAttention(
            inputLayerNorm(input),
            attentionMask: attentionMask,
            ssmMask: ssmMask,
            cache: cache
        )
        return hidden + feedForward(postAttentionLayerNorm(hidden))
    }
}

private final class KimiLinearBackbone: Module {
    let layerPlan: KimiLinearLayerPlan

    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    @ModuleInfo var layers: [KimiLinearDecoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: KimiLinearConfiguration) {
        let plan = KimiLinearLayerPlan(config)
        self.layerPlan = plan
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            KimiLinearDecoderLayer(config, layerIndex: layerIndex, layerPlan: plan)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let index = layerPlan.firstAttentionLayerIndex {
            attentionMask = createAttentionMask(
                h: hidden,
                cache: cacheValue(in: cache, at: index),
                returnArray: true
            )
        } else {
            attentionMask = .none
        }

        let ssmMask: MLXArray?
        if let index = layerPlan.firstLinearLayerIndex {
            ssmMask = createSSMMask(h: hidden, cache: cacheValue(in: cache, at: index) as? MambaCache)
        } else {
            ssmMask = nil
        }

        for (layerIndex, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                attentionMask: layer.isLinear ? .none : attentionMask,
                ssmMask: layer.isLinear ? ssmMask : nil,
                cache: cacheValue(in: cache, at: layerIndex)
            )
        }

        return hidden
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        norm(hiddenStates(inputs, cache: cache))
    }

    private func cacheValue(in cache: [KVCache]?, at index: Int) -> KVCache? {
        guard let cache, cache.indices.contains(index) else { return nil }
        return cache[index]
    }
}

internal final class KimiLinearModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let modelType: String

    private let config: KimiLinearConfiguration
    private let layerPlan: KimiLinearLayerPlan

    @ModuleInfo(key: "model") private var model: KimiLinearBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: KimiLinearConfiguration) {
        let plan = KimiLinearLayerPlan(config)
        self.config = config
        self.layerPlan = plan
        self.vocabularySize = config.vocabularySize
        self.modelType = config.modelType
        self.kvHeads = (0 ..< config.hiddenLayers).map { layerIndex in
            plan.isLinearLayer(layerIndex) ? config.linearAttention.numHeads : config.attentionHeads
        }
        _model.wrappedValue = KimiLinearBackbone(config)
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
        let hidden = lastTokenHiddenState(model.hiddenStates(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hidden), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { layerIndex in
            layerPlan.isLinearLayer(layerIndex)
                ? MambaCache(size: 4) as KVCache
                : KVCacheSimple() as KVCache
        }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights.filter { !$0.key.hasPrefix("model.mtp") },
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }

        for layerIndex in 0 ..< config.hiddenLayers {
            let layerPrefix = "model.layers.\(layerIndex)"
            if layerPlan.usesSparseExperts(config, layerIndex: layerIndex) {
                packSparseExperts(layerPrefix: layerPrefix, into: &sanitized)
            }
            if layerPlan.isLinearLayer(layerIndex) {
                sanitizeDeltaAttention(layerPrefix: layerPrefix, into: &sanitized)
            } else {
                splitKVProjection(prefix: "\(layerPrefix).self_attn", into: &sanitized)
            }
        }

        return sanitized.filter { key, _ in
            !key.contains(".block_sparse_moe.experts.")
                && !key.contains(".rotary_emb.inv_freq")
        }
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, $0.selfAttention.loraTargets) }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }

    private func packSparseExperts(layerPrefix: String, into weights: inout [String: MLXArray]) {
        let sourcePrefix = "\(layerPrefix).block_sparse_moe"
        let destinationPrefix = "\(layerPrefix).mlp"
        for (source, destination) in [
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj")
        ] {
            for tensorName in ["weight", "scales", "biases"] {
                let firstKey = "\(sourcePrefix).experts.0.\(source).\(tensorName)"
                guard weights[firstKey] != nil else { continue }
                let tensors = (0 ..< config.numExperts).map { expertIndex in
                    weights.removeValue(
                        forKey: "\(sourcePrefix).experts.\(expertIndex).\(source).\(tensorName)"
                    )!
                }
                weights["\(destinationPrefix).switch_mlp.\(destination).\(tensorName)"] =
                    MLX.stacked(tensors)
            }
        }

        for name in ["gate_proj", "up_proj", "down_proj"] {
            let sourceKey = "\(sourcePrefix).shared_experts.\(name).weight"
            if let value = weights.removeValue(forKey: sourceKey) {
                weights["\(destinationPrefix).shared_experts.\(name).weight"] = value
            }
        }

        if let gate = weights.removeValue(forKey: "\(sourcePrefix).gate.weight") {
            weights["\(destinationPrefix).gate.weight"] = gate
        }
        if let bias = weights.removeValue(forKey: "\(sourcePrefix).gate.e_score_correction_bias") {
            weights["\(destinationPrefix).e_score_correction_bias"] = bias
        }
    }

    private func sanitizeDeltaAttention(layerPrefix: String, into weights: inout [String: MLXArray]) {
        let attentionPrefix = "\(layerPrefix).self_attn"
        for (sourceName, destinationName) in [
            ("q_conv1d", "q_conv"),
            ("k_conv1d", "k_conv"),
            ("v_conv1d", "v_conv")
        ] {
            let sourceKey = "\(attentionPrefix).\(sourceName).weight"
            if let value = weights.removeValue(forKey: sourceKey) {
                weights["\(attentionPrefix).\(destinationName).conv.weight"] =
                    value.ndim == 3 ? value.movedAxis(source: 2, destination: 1) : value
            }
        }

        let dtKey = "\(attentionPrefix).dt_bias"
        if let value = weights[dtKey], value.ndim > 1 {
            weights[dtKey] = value.reshaped(-1)
        }
        let aLogKey = "\(attentionPrefix).A_log"
        if let value = weights[aLogKey], value.ndim > 1 {
            weights[aLogKey] = value.reshaped(-1)
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
        ).split(weight: projection)

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
