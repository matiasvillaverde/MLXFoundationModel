import Foundation
import MLX
import MLXNN

internal enum LFM2MoELayerKind: String, Codable, Sendable, Equatable {
    case convolution = "conv"
    case fullAttention = "full_attention"
}

internal struct LFM2MoELayerPlan: Sendable, Equatable {
    internal let kinds: [LFM2MoELayerKind]
    internal let firstAttentionIndex: Int?
    internal let firstConvolutionIndex: Int?

    internal init(kinds: [LFM2MoELayerKind]) {
        self.kinds = kinds
        self.firstAttentionIndex = kinds.firstIndex(of: .fullAttention)
        self.firstConvolutionIndex = kinds.firstIndex(of: .convolution)
    }

    internal var attentionIndices: [Int] {
        kinds.enumerated().compactMap { index, kind in
            kind == .fullAttention ? index : nil
        }
    }

    internal func usesAttention(layerIndex: Int) -> Bool {
        kinds[layerIndex] == .fullAttention
    }
}

internal struct LFM2MoEAttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let queryDimensions: Int
    internal let keyValueDimensions: Int
    internal let scale: Float

    internal init(_ configuration: LFM2MoEConfiguration) {
        precondition(configuration.hiddenSize > 0, "hidden_size must be positive")
        precondition(configuration.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(configuration.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(
            configuration.hiddenSize % configuration.attentionHeads == 0,
            "hidden_size must be divisible by num_attention_heads"
        )

        self.hiddenSize = configuration.hiddenSize
        self.attentionHeads = configuration.attentionHeads
        self.keyValueHeads = configuration.kvHeads
        self.headDimensions = configuration.hiddenSize / configuration.attentionHeads
        self.queryDimensions = attentionHeads * headDimensions
        self.keyValueDimensions = keyValueHeads * headDimensions
        self.scale = pow(Float(headDimensions), -0.5)
    }
}

internal struct LFM2MoEConvolutionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let stateLength: Int
    internal let projectionDimensions: Int
    internal let bias: Bool

    internal init(_ configuration: LFM2MoEConfiguration) {
        precondition(configuration.hiddenSize > 0, "hidden_size must be positive")
        precondition(configuration.convLCache > 1, "conv_L_cache must exceed one")
        self.hiddenSize = configuration.hiddenSize
        self.stateLength = configuration.convLCache - 1
        self.projectionDimensions = configuration.hiddenSize * 3
        self.bias = configuration.convBias
    }
}

private struct LFM2MoERouterPlan: Sendable, Equatable {
    let topK: Int
    let normalizesTopK: Bool
    let usesExpertBias: Bool
    let scalingFactor: Float

    init(_ configuration: LFM2MoEConfiguration) {
        precondition(configuration.numExperts > 0, "num_experts must be positive")
        precondition(configuration.numExpertsPerToken > 0, "num_experts_per_tok must be positive")
        precondition(
            configuration.numExpertsPerToken <= configuration.numExperts,
            "num_experts_per_tok must be less than or equal to num_experts"
        )
        self.topK = configuration.numExpertsPerToken
        self.normalizesTopK = configuration.normTopkProb
        self.usesExpertBias = configuration.useExpertBias
        self.scalingFactor = configuration.routedScalingFactor
    }
}

internal struct LFM2MoEConfiguration: Codable, Sendable {
    internal let modelType: String
    internal let vocabularySize: Int
    internal let hiddenSize: Int
    internal let intermediateSize: Int
    internal let moeIntermediateSize: Int
    internal let hiddenLayers: Int
    internal let numExperts: Int
    internal let numExpertsPerToken: Int
    internal let normTopkProb: Bool
    internal let attentionHeads: Int
    internal let kvHeads: Int
    internal let maxPositionEmbeddings: Int
    internal let useExpertBias: Bool
    internal let numDenseLayers: Int
    internal let normEps: Float
    internal let convBias: Bool
    internal let convLCache: Int
    internal let ropeTheta: Float
    internal let routedScalingFactor: Float

    private let _fullAttnIdxs: [Int]?
    private let layerTypes: [LFM2MoELayerKind]?
    private let ropeParameters: [String: StringOrNumber]?

    internal var fullAttnIdxs: [Int] {
        layerPlan.attentionIndices
    }

    internal var layerPlan: LFM2MoELayerPlan {
        if let fullAttentionIndices = _fullAttnIdxs {
            let attentionSet = Set(fullAttentionIndices)
            return LFM2MoELayerPlan(kinds: (0 ..< hiddenLayers).map { layerIndex in
                attentionSet.contains(layerIndex) ? .fullAttention : .convolution
            })
        }
        if let layerTypes {
            return LFM2MoELayerPlan(kinds: layerTypes)
        }
        return LFM2MoELayerPlan(kinds: Array(repeating: .convolution, count: hiddenLayers))
    }

    internal var headDimensions: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case useExpertBias = "use_expert_bias"
        case numDenseLayers = "num_dense_layers"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case ropeTheta = "rope_theta"
        case routedScalingFactor = "routed_scaling_factor"
        case ropeParameters = "rope_parameters"
        case _fullAttnIdxs = "full_attn_idxs"
        case layerTypes = "layer_types"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.useExpertBias = try container.decode(Bool.self, forKey: .useExpertBias)
        self.numDenseLayers = try container.decode(Int.self, forKey: .numDenseLayers)
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.convBias = try container.decode(Bool.self, forKey: .convBias)
        self.convLCache = try container.decode(Int.self, forKey: .convLCache)
        self.routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        self.ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        self.ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? ropeTheta
        self._fullAttnIdxs = try container.decodeIfPresent([Int].self, forKey: ._fullAttnIdxs)
        if let layerTypes = try container.decodeIfPresent(
            [LFM2MoELayerKind].self,
            forKey: .layerTypes
        ) {
            guard layerTypes.count == self.hiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes,
                    in: container,
                    debugDescription: "layer_types count must match num_hidden_layers"
                )
            }
            self.layerTypes = layerTypes
        } else {
            self.layerTypes = nil
        }
    }
}

class LFM2MoEAttention: Module {
    let layout: LFM2MoEAttentionLayout

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

    let rope: RoPE

    init(_ args: LFM2MoEConfiguration) {
        self.layout = LFM2MoEAttentionLayout(args)

        _qProj.wrappedValue = Linear(layout.hiddenSize, layout.queryDimensions, bias: false)
        _kProj.wrappedValue = Linear(layout.hiddenSize, layout.keyValueDimensions, bias: false)
        _vProj.wrappedValue = Linear(layout.hiddenSize, layout.keyValueDimensions, bias: false)
        _outProj.wrappedValue = Linear(layout.queryDimensions, layout.hiddenSize, bias: false)

        _qLayerNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: args.normEps)
        _kLayerNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: args.normEps)

        self.rope = RoPE(
            dimensions: layout.headDimensions,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = qLayerNorm(queries.reshaped(B, L, layout.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        keys = kLayerNorm(keys.reshaped(B, L, layout.keyValueHeads, -1))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, layout.keyValueHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outProj(output)
    }
}

class LFM2MoEShortConv: Module {
    let layout: LFM2MoEConvolutionLayout
    let layerIdx: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM2MoEConfiguration, layerIdx: Int) {
        self.layout = LFM2MoEConvolutionLayout(args)
        self.layerIdx = layerIdx

        _conv.wrappedValue = Conv1d(
            inputChannels: layout.hiddenSize,
            outputChannels: layout.hiddenSize,
            kernelSize: layout.stateLength + 1,
            groups: layout.hiddenSize,
            bias: layout.bias
        )

        _inProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.projectionDimensions,
            bias: layout.bias
        )
        _outProj.wrappedValue = Linear(layout.hiddenSize, layout.hiddenSize, bias: layout.bias)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let BCx = inProj(x)
        let parts = BCx.split(parts: 3, axis: -1)
        let B = parts[0]
        let C = parts[1]
        let xComp = parts[2]
        var Bx = B * xComp

        if let mask {
            let expandedMask = mask[.ellipsis, .newAxis]
            let zeros = MLXArray.zeros(Bx.shape, dtype: Bx.dtype)
            Bx = MLX.where(expandedMask, Bx, zeros)
        }

        var state = cache?[0]
        if state == nil {
            state = MLXArray.zeros(
                [Bx.dim(0), layout.stateLength, layout.hiddenSize],
                dtype: Bx.dtype
            )
        }

        Bx = concatenated([state!, Bx], axis: -2)
        if let cache {
            let start = Bx.dim(1) - layout.stateLength
            cache[0] = Bx[0..., start..., 0...]
        }

        let convOut = conv(Bx)
        let y = C * convOut
        return outProj(y)
    }
}

class LFM2MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: LFM2MoEConfiguration, intermediateSize: Int? = nil) {
        let hidden = args.hiddenSize
        let inter = intermediateSize ?? args.intermediateSize

        _gateProj.wrappedValue = Linear(hidden, inter, bias: false)
        _upProj.wrappedValue = Linear(hidden, inter, bias: false)
        _downProj.wrappedValue = Linear(inter, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class LFM2MoESparseBlock: Module, UnaryLayer {
    let numExperts: Int
    fileprivate let routerPlan: LFM2MoERouterPlan

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray?

    init(_ args: LFM2MoEConfiguration) {
        self.numExperts = args.numExperts
        self.routerPlan = LFM2MoERouterPlan(args)

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: numExperts,
            bias: false
        )

        if routerPlan.usesExpertBias {
            _expertBias.wrappedValue = MLXArray.zeros([numExperts])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = lfm2MoERouter(
            logits: gate(x),
            expertBias: expertBias,
            topK: routerPlan.topK,
            normTopKProb: routerPlan.normalizesTopK,
            useExpertBias: routerPlan.usesExpertBias,
            routedScalingFactor: routerPlan.scalingFactor
        )

        let expertOutputs = switchMLP(x, indices)
        let weighted = expertOutputs * scores.asType(x.dtype)[.ellipsis, .newAxis]
        return weighted.sum(axis: -2)
    }
}

internal func lfm2MoERouter(
    logits: MLXArray,
    expertBias: MLXArray?,
    topK: Int,
    normTopKProb: Bool,
    useExpertBias: Bool,
    routedScalingFactor: Float
) -> (indices: MLXArray, scores: MLXArray) {
    let routingWeights = sigmoid(logits.asType(.float32))
    let selectionScores: MLXArray
    if useExpertBias, let expertBias {
        selectionScores = routingWeights + expertBias
    } else {
        selectionScores = routingWeights
    }

    let indices = argPartition(-selectionScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var scores = takeAlong(routingWeights, indices, axis: -1)
    if normTopKProb {
        let denominator = scores.sum(axis: -1, keepDims: true) + 1e-6
        scores = scores / denominator
    }
    scores = scores * routedScalingFactor
    return (indices, scores)
}

class LFM2MoEDecoderLayer: Module {
    let layerKind: LFM2MoELayerKind
    let usesDenseFeedForward: Bool

    @ModuleInfo(key: "self_attn") var attention: LFM2MoEAttention?
    @ModuleInfo(key: "conv") var conv: LFM2MoEShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ args: LFM2MoEConfiguration, layerIdx: Int) {
        self.layerKind = args.layerPlan.kinds[layerIdx]
        self.usesDenseFeedForward = layerIdx < args.numDenseLayers

        if layerKind == .fullAttention {
            _attention.wrappedValue = LFM2MoEAttention(args)
        } else {
            _conv.wrappedValue = LFM2MoEShortConv(args, layerIdx: layerIdx)
        }

        if usesDenseFeedForward {
            _feedForward.wrappedValue = LFM2MoEMLP(args)
        } else {
            _feedForward.wrappedValue = LFM2MoESparseBlock(args)
        }

        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    var isAttentionLayer: Bool {
        layerKind == .fullAttention
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let residual = operatorNorm(x)
        let r: MLXArray
        if isAttentionLayer {
            guard let attention else {
                preconditionFailure("LFM2 MoE attention layer is missing its attention module")
            }
            r = attention(residual, mask: attentionMask, cache: cache)
        } else {
            guard let conv else {
                preconditionFailure("LFM2 MoE convolution layer is missing its convolution module")
            }
            r = conv(residual, mask: ssmMask, cache: cache as? MambaCache)
        }

        let h = x + r
        let out = feedForward(ffnNorm(h))
        return h + out
    }
}

internal class LFM2MoEModelInner: Module {
    let layers: [LFM2MoEDecoderLayer]
    let layerPlan: LFM2MoELayerPlan

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ args: LFM2MoEConfiguration) {
        self.layerPlan = args.layerPlan
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { LFM2MoEDecoderLayer(args, layerIdx: $0) }

        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hidden = inputEmbeddings ?? embedTokens(inputs)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let index = layerPlan.firstAttentionIndex,
                let cache,
                index < cache.count
            else { return .none }
            return createAttentionMask(h: hidden, cache: cache[index])
        }()

        let ssmMask: MLXArray? = {
            guard let index = layerPlan.firstConvolutionIndex,
                let cache,
                index < cache.count
            else { return nil }
            return createSSMMask(h: hidden, cache: cache[index] as? MambaCache)
        }()

        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache?[i])
        }

        return embeddingNorm(hidden)
    }
}

private struct LFM2MoEWeightSanitizer {
    let configuration: LFM2MoEConfiguration

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        packExperts(in: normalizeProjectionNames(weights))
    }

    private func normalizeProjectionNames(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        sanitized.reserveCapacity(weights.count)

        for (name, param) in weights {
            sanitized[renamedProjectionKey(name)] = normalizedTensor(param, key: name)
        }

        return sanitized
    }

    private func normalizedTensor(_ tensor: MLXArray, key: String) -> MLXArray {
        if key.contains("conv.weight"), tensor.dim(-1) > tensor.dim(1) {
            return tensor.transposed(0, 2, 1)
        }
        return tensor
    }

    private func renamedProjectionKey(_ key: String) -> String {
        let replacements = [
            "w1.weight": "gate_proj.weight",
            "w2.weight": "down_proj.weight",
            "w3.weight": "up_proj.weight",
        ]
        var updated = key
        for (old, new) in replacements where updated.contains(old) {
            updated = updated.replacingOccurrences(of: old, with: new)
        }
        return updated
    }

    private func packExperts(in weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights

        for layerIdx in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIdx)"
            let expertPrefix = "\(prefix).feed_forward.experts"
            guard packed["\(expertPrefix).0.gate_proj.weight"] != nil else {
                continue
            }

            for name in ["gate_proj", "down_proj", "up_proj"] {
                packExpertProjection(name, expertPrefix: expertPrefix, layerPrefix: prefix, in: &packed)
            }
        }

        return packed
    }

    private func packExpertProjection(
        _ name: String,
        expertPrefix: String,
        layerPrefix: String,
        in weights: inout [String: MLXArray]
    ) {
        let firstKey = "\(expertPrefix).0.\(name).weight"
        guard weights[firstKey] != nil else {
            return
        }
        let keys = (0 ..< configuration.numExperts).map { expert in
            "\(expertPrefix).\(expert).\(name).weight"
        }
        guard keys.allSatisfy({ weights[$0] != nil }) else {
            return
        }
        let stacked = keys.compactMap { weights.removeValue(forKey: $0) }
        weights["\(layerPrefix).feed_forward.switch_mlp.\(name).weight"] = MLX.stacked(stacked)
    }
}

internal class LFM2MoEModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    let configuration: LFM2MoEConfiguration

    @ModuleInfo(key: "model") fileprivate var model: LFM2MoEModelInner

    internal init(_ args: LFM2MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        let layerPlan = args.layerPlan
        self.kvHeads = (0 ..< args.hiddenLayers).map { layerIdx in
            layerPlan.usesAttention(layerIndex: layerIdx) ? args.kvHeads : 0
        }
        self._model.wrappedValue = LFM2MoEModelInner(args)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: model.embedTokens.asLinear(hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        LFM2MoEWeightSanitizer(configuration: configuration).sanitize(weights)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let layerPlan = configuration.layerPlan
        return (0 ..< configuration.hiddenLayers).map { layerIdx in
            if layerPlan.usesAttention(layerIndex: layerIdx) {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }
}

extension LFM2MoEModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.compactMap { layer in
            guard layer.isAttentionLayer, let attention = layer.attention else {
                return nil
            }
            return (attention, ["q_proj", "v_proj"])
        }
    }
}
