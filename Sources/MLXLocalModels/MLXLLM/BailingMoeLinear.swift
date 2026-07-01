import Foundation
import MLX
import MLXFast
import MLXNN

internal struct BailingMoeLinearConfiguration: Codable, Equatable, Sendable {
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
    var layerGroupSize: Int
    var groupNormSize: Int
    var ropeScaling: [String: StringOrNumber]?
    var ropeTraditional: Bool
    var useBias: Bool
    var useQKVBias: Bool
    var normHead: Bool
    var normSoftmax: Bool
    var useQKNorm: Bool
    var tieWordEmbeddings: Bool
    var partialRotaryFactor: Float
    var moeRouterEnableExpertBias: Bool
    var moeRouterEnableRoutedScaling: Bool
    var routedScalingFactor: Float
    var scoreFunction: String
    var nGroup: Int
    var topkGroup: Int
    var useRMSNorm: Bool
    var moeSharedExpertIntermediateSize: Int?
    var moeRouterEnableSharedExpert: Bool
    var headDim: Int?

    internal init(
        modelType: String = "bailing_moe_linear",
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
        layerGroupSize: Int = 4,
        groupNormSize: Int = 1,
        ropeScaling: [String: StringOrNumber]? = nil,
        ropeTraditional: Bool = false,
        useBias: Bool = false,
        useQKVBias: Bool = false,
        normHead: Bool = false,
        normSoftmax: Bool = false,
        useQKNorm: Bool = false,
        tieWordEmbeddings: Bool = false,
        partialRotaryFactor: Float = 1,
        moeRouterEnableExpertBias: Bool = false,
        moeRouterEnableRoutedScaling: Bool = true,
        routedScalingFactor: Float = 1,
        scoreFunction: String = "softmax",
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        useRMSNorm: Bool = true,
        moeSharedExpertIntermediateSize: Int? = nil,
        moeRouterEnableSharedExpert: Bool = true,
        headDim: Int? = nil
    ) {
        precondition(hiddenSize > 0, "Bailing linear hidden_size must be positive")
        precondition(intermediateSize > 0, "Bailing linear intermediate_size must be positive")
        precondition(attentionHeads > 0, "Bailing linear attention heads must be positive")
        precondition(hiddenLayers > 0, "Bailing linear hidden layer count must be positive")
        precondition(vocabularySize > 0, "Bailing linear vocabulary size must be positive")
        precondition(layerGroupSize > 0, "Bailing linear layer_group_size must be positive")
        precondition(groupNormSize > 0, "Bailing linear group_norm_size must be positive")

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
        self.layerGroupSize = layerGroupSize
        self.groupNormSize = groupNormSize
        self.ropeScaling = ropeScaling
        self.ropeTraditional = ropeTraditional
        self.useBias = useBias
        self.useQKVBias = useQKVBias
        self.normHead = normHead
        self.normSoftmax = normSoftmax
        self.useQKNorm = useQKNorm
        self.tieWordEmbeddings = tieWordEmbeddings
        self.partialRotaryFactor = partialRotaryFactor
        self.moeRouterEnableExpertBias = moeRouterEnableExpertBias
        self.moeRouterEnableRoutedScaling = moeRouterEnableRoutedScaling
        self.routedScalingFactor = routedScalingFactor
        self.scoreFunction = scoreFunction
        self.nGroup = nGroup
        self.topkGroup = topkGroup ?? nGroup
        self.useRMSNorm = useRMSNorm
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.moeRouterEnableSharedExpert = moeRouterEnableSharedExpert
        self.headDim = headDim
    }

    var resolvedHeadDim: Int {
        headDim ?? hiddenSize / attentionHeads
    }

    var moeBridgeConfiguration: BailingMoeConfiguration {
        BailingMoeConfiguration(
            modelType: modelType,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            maxPositionEmbeddings: maxPositionEmbeddings,
            moeIntermediateSize: moeIntermediateSize,
            numExperts: max(numExperts, 1),
            numSharedExperts: moeRouterEnableSharedExpert ? numSharedExperts : 0,
            normTopkProb: normTopkProb,
            attentionHeads: attentionHeads,
            numExpertsPerToken: numExpertsPerToken,
            hiddenLayers: hiddenLayers,
            kvHeads: kvHeads,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            vocabularySize: vocabularySize,
            firstKDenseReplace: firstKDenseReplace,
            ropeScaling: ropeScaling,
            useBias: useBias,
            useQKVBias: useQKVBias,
            useQKNorm: useQKNorm,
            tieWordEmbeddings: tieWordEmbeddings,
            partialRotaryFactor: partialRotaryFactor,
            moeRouterEnableExpertBias: moeRouterEnableExpertBias,
            routedScalingFactor: routedScalingFactor,
            scoreFunction: scoreFunction,
            nGroup: nGroup,
            topkGroup: topkGroup,
            moeSharedExpertIntermediateSize: moeSharedExpertIntermediateSize
        )
    }

    private enum CodingKeys: String, CodingKey {
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
        case layerGroupSize = "layer_group_size"
        case groupNormSize = "group_norm_size"
        case ropeScaling = "rope_scaling"
        case ropeTraditional = "rope_traditional"
        case useBias = "use_bias"
        case useQKVBias = "use_qkv_bias"
        case normHead = "norm_head"
        case normSoftmax = "norm_softmax"
        case useQKNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case moeRouterEnableRoutedScaling = "moe_router_enable_routed_scaling"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunction = "score_function"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case useRMSNorm = "use_rmsnorm"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case moeRouterEnableSharedExpert = "moe_router_enable_shared_expert"
        case headDim = "head_dim"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "bailing_moe_linear",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ),
            numExperts: try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0,
            numSharedExperts: try container.decodeIfPresent(Int.self, forKey: .numSharedExperts)
                ?? 0,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false,
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            numExpertsPerToken: try container.decodeIfPresent(
                Int.self,
                forKey: .numExpertsPerToken
            ) ?? 1,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            firstKDenseReplace: try container.decodeIfPresent(
                Int.self,
                forKey: .firstKDenseReplace
            ) ?? 0,
            layerGroupSize: try container.decodeIfPresent(Int.self, forKey: .layerGroupSize) ?? 4,
            groupNormSize: try container.decodeIfPresent(Int.self, forKey: .groupNormSize) ?? 1,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                ?? false,
            useBias: try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            useQKVBias: try container.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false,
            normHead: try container.decodeIfPresent(Bool.self, forKey: .normHead) ?? false,
            normSoftmax: try container.decodeIfPresent(Bool.self, forKey: .normSoftmax) ?? false,
            useQKNorm: try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 1,
            moeRouterEnableExpertBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .moeRouterEnableExpertBias
            ) ?? false,
            moeRouterEnableRoutedScaling: try container.decodeIfPresent(
                Bool.self,
                forKey: .moeRouterEnableRoutedScaling
            ) ?? true,
            routedScalingFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .routedScalingFactor
            ) ?? 1,
            scoreFunction: try container.decodeIfPresent(String.self, forKey: .scoreFunction)
                ?? "softmax",
            nGroup: try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1,
            topkGroup: try container.decodeIfPresent(Int.self, forKey: .topkGroup),
            useRMSNorm: try container.decodeIfPresent(Bool.self, forKey: .useRMSNorm) ?? true,
            moeSharedExpertIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeSharedExpertIntermediateSize
            ),
            moeRouterEnableSharedExpert: try container.decodeIfPresent(
                Bool.self,
                forKey: .moeRouterEnableSharedExpert
            ) ?? true,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim)
        )
    }
}

internal struct BailingMoeLinearLayerPlan: Equatable, Sendable {
    let isGlobal: [Bool]
    let firstLinearLayerIndex: Int?
    let firstGlobalLayerIndex: Int?

    init(_ config: BailingMoeLinearConfiguration) {
        self.isGlobal = (0 ..< config.hiddenLayers).map { layerIndex in
            (layerIndex + 1) % config.layerGroupSize == 0
                || layerIndex >= (config.hiddenLayers / config.layerGroupSize)
                    * config.layerGroupSize
        }
        self.firstLinearLayerIndex = isGlobal.firstIndex(of: false)
        self.firstGlobalLayerIndex = isGlobal.firstIndex(of: true)
    }

    func isGlobalLayer(_ layerIndex: Int) -> Bool {
        isGlobal[layerIndex]
    }

    func usesSparseExperts(_ config: BailingMoeLinearConfiguration, layerIndex: Int) -> Bool {
        config.numExperts > 0 && layerIndex >= config.firstKDenseReplace
    }
}

private struct BailingMoeLinearAttentionLayout: Sendable, Equatable {
    let hiddenSize: Int
    let attentionHeads: Int
    let keyValueHeads: Int
    let headDim: Int
    let rotaryDimensions: Int
    let attentionScale: Float
    let queryProjectionSize: Int
    let keyValueProjectionSize: Int
    let queryKeyValueProjectionSize: Int

    init(_ config: BailingMoeLinearConfiguration, keyValueHeads: Int) {
        let headDim = config.resolvedHeadDim
        precondition(headDim > 0, "Bailing linear head_dim must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: keyValueHeads),
            "Bailing linear attention heads must group KV heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = keyValueHeads
        self.headDim = headDim
        self.rotaryDimensions = max(1, Int(Float(headDim) * config.partialRotaryFactor))
        self.attentionScale = pow(Float(headDim), -0.5)
        self.queryProjectionSize = config.attentionHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.queryKeyValueProjectionSize = queryProjectionSize + 2 * keyValueProjectionSize
    }
}

private final class BailingMoeLinearGroupRMSNorm: Module, UnaryLayer {
    private let groups: Int
    private let eps: Float

    @ParameterInfo(key: "weight") private var weight: MLXArray

    init(dimensions: Int, groups: Int, eps: Float) {
        precondition(dimensions.isMultiple(of: groups), "Group RMSNorm dimensions must divide")
        self.groups = groups
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let grouped = unflatten(input, axis: -1, shape: [groups, -1])
        let normalized = MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
        return weight * flattened(normalized, start: -2, end: -1)
    }
}

private final class BailingMoeLinearGlobalAttention: Module {
    let layout: BailingMoeLinearAttentionLayout

    @ModuleInfo(key: "query_key_value") private var queryKeyValue: Linear
    @ModuleInfo(key: "dense") private var output: Linear
    @ModuleInfo(key: "query_layernorm") private var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") private var keyNorm: RMSNorm?

    private let rope: RoPELayer

    init(_ config: BailingMoeLinearConfiguration) {
        self.layout = BailingMoeLinearAttentionLayout(config, keyValueHeads: config.kvHeads)
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
            _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
        }
        self.rope = initializeRope(
            dims: layout.rotaryDimensions,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
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
        let tokenCount = input.dim(1)
        let projections = split(
            queryKeyValue(input),
            indices: [
                layout.queryProjectionSize,
                layout.queryProjectionSize + layout.keyValueProjectionSize
            ],
            axis: -1
        )

        var queries = projections[0]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDim)
        var keys = projections[1]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
        let values = projections[2]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)

        if let queryNorm {
            queries = queryNorm(queries)
        }
        if let keyNorm {
            keys = keyNorm(keys)
        }

        queries = rope(queries.transposed(0, 2, 1, 3), offset: cache?.offset ?? 0)
        keys = rope(keys.transposed(0, 2, 1, 3), offset: cache?.offset ?? 0)

        return output(
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.attentionScale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, -1)
        )
    }
}

private func bailingMoeRecurrentGLA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    slope: MLXArray,
    scale: Float,
    state: MLXArray?
) -> (MLXArray, MLXArray) {
    let tokenCount = queries.dim(2)
    let decay = exp(slope).reshaped(1, slope.dim(0), 1, 1).asType(queries.dtype)
    var runningState = state
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(tokenCount)

    let scaledQueries = queries * scale
    for tokenIndex in 0 ..< tokenCount {
        let query = scaledQueries[0..., 0..., tokenIndex ..< tokenIndex + 1, 0...]
        let key = keys[0..., 0..., tokenIndex ..< tokenIndex + 1, 0...]
        let value = values[0..., 0..., tokenIndex ..< tokenIndex + 1, 0...]
        let update = key.transposed(0, 1, 3, 2).matmul(value)
        if let state = runningState {
            runningState = state * decay + update
        } else {
            runningState = update
        }
        outputs.append(query.matmul(runningState!))
    }

    return (concatenated(outputs, axis: 2), runningState!)
}

private final class BailingMoeLinearAttention: Module {
    let layout: BailingMoeLinearAttentionLayout
    private let slope: MLXArray
    private let rope: RoPELayer

    @ModuleInfo(key: "query_key_value") private var queryKeyValue: Linear
    @ModuleInfo(key: "dense") private var output: Linear
    @ModuleInfo(key: "g_proj") private var gateProjection: Linear
    @ModuleInfo(key: "g_norm") private var gateNorm: BailingMoeLinearGroupRMSNorm
    @ModuleInfo(key: "query_layernorm") private var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") private var keyNorm: RMSNorm?

    init(_ config: BailingMoeLinearConfiguration, layerIndex: Int) {
        self.layout = BailingMoeLinearAttentionLayout(
            config,
            keyValueHeads: config.attentionHeads
        )
        self.slope = Self.slopes(
            headCount: config.attentionHeads,
            layerIndex: layerIndex,
            layerCount: config.hiddenLayers
        )
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
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        _gateNorm.wrappedValue = BailingMoeLinearGroupRMSNorm(
            dimensions: layout.queryProjectionSize,
            groups: config.groupNormSize,
            eps: config.rmsNormEps
        )
        if config.useQKNorm {
            _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: layout.headDim, eps: config.rmsNormEps)
        }
        self.rope = initializeRope(
            dims: layout.rotaryDimensions,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    func callAsFunction(_ input: MLXArray, cache: MambaCache?, offset: Int) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)
        let qkv = queryKeyValue(input)
            .reshaped(
                batchSize,
                tokenCount,
                layout.attentionHeads + 2 * layout.keyValueHeads,
                layout.headDim
            )
        let parts = split(
            qkv,
            indices: [
                layout.attentionHeads,
                layout.attentionHeads + layout.keyValueHeads
            ],
            axis: 2
        )

        var queries = parts[0]
        var keys = parts[1]
        let values = parts[2].transposed(0, 2, 1, 3)
        if let queryNorm {
            queries = queryNorm(queries)
        }
        if let keyNorm {
            keys = keyNorm(keys)
        }

        queries = rope(queries.transposed(0, 2, 1, 3), offset: offset)
        keys = rope(keys.transposed(0, 2, 1, 3), offset: offset)

        let (attentionOutput, nextState) = bailingMoeRecurrentGLA(
            queries: queries,
            keys: keys,
            values: values,
            slope: slope,
            scale: layout.attentionScale,
            state: cache?[0]
        )
        if let cache {
            cache[0] = nextState
            cache.offset += tokenCount
        }

        let projected = attentionOutput
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, -1)
        let gated = gateNorm(projected) * sigmoid(gateProjection(input))
        return output(gated)
    }

    private static func slopes(headCount: Int, layerIndex: Int, layerCount: Int) -> MLXArray {
        func powerOfTwoSlopes(_ n: Int) -> [Float] {
            let exponentBase = pow(2, -(pow(2, -(log2(Float(n)) - 3))))
            return (1 ... n).map { pow(exponentBase, Float($0)) }
        }

        let values: [Float]
        if headCount > 0 && (headCount & (headCount - 1)) == 0 {
            values = powerOfTwoSlopes(headCount)
        } else {
            let lowerPower = 1 << Int(floor(log2(Float(headCount))))
            let extra = powerOfTwoSlopes(2 * lowerPower)
                .enumerated()
                .compactMap { index, value in index.isMultiple(of: 2) ? value : nil }
                .prefix(headCount - lowerPower)
            values = powerOfTwoSlopes(lowerPower) + Array(extra)
        }

        let denominator = max(1, layerCount - 1)
        let layerPosition = max(0, layerIndex - 1)
        let layerFactor = 1 - (Float(layerPosition) / Float(denominator)) + 1e-5
        return MLXArray(values.map { -$0 * layerFactor })
    }
}

private final class BailingMoeLinearBlock: Module {
    let isGlobal: Bool

    @ModuleInfo(key: "attention") fileprivate var globalAttention: BailingMoeLinearGlobalAttention?
    @ModuleInfo(key: "attention") fileprivate var linearAttention: BailingMoeLinearAttention?
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(
        _ config: BailingMoeLinearConfiguration,
        layerIndex: Int,
        layerPlan: BailingMoeLinearLayerPlan
    ) {
        self.isGlobal = layerPlan.isGlobalLayer(layerIndex)
        if isGlobal {
            _globalAttention.wrappedValue = BailingMoeLinearGlobalAttention(config)
        } else {
            _linearAttention.wrappedValue = BailingMoeLinearAttention(config, layerIndex: layerIndex)
        }

        if layerPlan.usesSparseExperts(config, layerIndex: layerIndex) {
            _feedForward.wrappedValue = BailingMoeSparseBlock(config.moeBridgeConfiguration)
        } else {
            _feedForward.wrappedValue = BailingMoeMLP(config.moeBridgeConfiguration)
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
        cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        let normalized = inputLayerNorm(input)
        let attentionOutput: MLXArray
        if isGlobal {
            guard let globalAttention else {
                preconditionFailure("Bailing linear global layer is missing attention")
            }
            attentionOutput = globalAttention(normalized, mask: attentionMask, cache: cache)
        } else {
            guard let linearAttention else {
                preconditionFailure("Bailing linear layer is missing GLA attention")
            }
            attentionOutput = linearAttention(normalized, cache: cache as? MambaCache, offset: offset)
        }

        let hidden = input + attentionOutput
        return hidden + feedForward(postAttentionLayerNorm(hidden))
    }
}

private final class BailingMoeLinearBackbone: Module {
    private let config: BailingMoeLinearConfiguration
    private let layerPlan: BailingMoeLinearLayerPlan

    @ModuleInfo(key: "word_embeddings") fileprivate var wordEmbeddings: Embedding
    @ModuleInfo var layers: [BailingMoeLinearBlock]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: BailingMoeLinearConfiguration) {
        let plan = BailingMoeLinearLayerPlan(config)
        self.config = config
        self.layerPlan = plan
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            BailingMoeLinearBlock(config, layerIndex: layerIndex, layerPlan: plan)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = wordEmbeddings(inputs)
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let index = layerPlan.firstGlobalLayerIndex {
                createAttentionMask(h: hidden, cache: cache?[index])
            } else {
                .none
            }
        let offset = layerPlan.firstGlobalLayerIndex
            .flatMap { cache?[$0].offset }
            ?? layerPlan.firstLinearLayerIndex.flatMap { cache?[$0].offset }
            ?? 0

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                attentionMask: layer.isGlobal ? attentionMask : .none,
                cache: cache?[index],
                offset: offset
            )
        }
        return norm(hidden)
    }
}

internal final class BailingMoeLinearModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let modelType: String

    private let config: BailingMoeLinearConfiguration
    private let layerPlan: BailingMoeLinearLayerPlan

    @ModuleInfo(key: "model") private var model: BailingMoeLinearBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: BailingMoeLinearConfiguration) {
        let plan = BailingMoeLinearLayerPlan(config)
        self.config = config
        self.layerPlan = plan
        self.vocabularySize = config.vocabularySize
        self.modelType = config.modelType
        self.kvHeads = (0 ..< config.hiddenLayers).map { layerIndex in
            plan.isGlobalLayer(layerIndex) ? config.kvHeads : config.attentionHeads
        }
        _model.wrappedValue = BailingMoeLinearBackbone(config)
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
        let hidden = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hidden), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { layerIndex in
            layerPlan.isGlobalLayer(layerIndex)
                ? KVCacheSimple() as KVCache
                : MambaCache() as KVCache
        }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        } else if config.normHead, let weight = sanitized["lm_head.weight"] {
            let dtype = weight.dtype
            let norm = sqrt(sum(square(weight.asType(.float32)), axis: 0, keepDims: true)) + 1e-7
            sanitized["lm_head.weight"] = (weight / norm.asType(dtype)).asType(dtype)
        }

        sanitized = sanitized.filter { !$0.key.contains("attention.rotary_emb.inv_freq") }
        sanitized = BailingMoeExpertPackingPlan(config.moeBridgeConfiguration).pack(sanitized)
        for layerIndex in config.firstKDenseReplace ..< config.hiddenLayers {
            let prefix = "model.layers.\(layerIndex).mlp.gate"
            if let weight = sanitized.removeValue(forKey: "\(prefix).weight") {
                sanitized["\(prefix).gate_proj.weight"] = weight
            }
            if let bias = sanitized.removeValue(forKey: "\(prefix).bias") {
                sanitized["\(prefix).gate_proj.bias"] = bias
            }
        }
        return sanitized
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.wordEmbeddings.asLinear(hiddenStates)
    }
}

extension BailingMoeLinearModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.compactMap { layer -> (Module, [String])? in
            if let attention = layer.globalAttention {
                return (attention, ["query_key_value"])
            }
            if let attention = layer.linearAttention {
                return (attention, ["query_key_value"])
            }
            return nil
        }
    }
}
