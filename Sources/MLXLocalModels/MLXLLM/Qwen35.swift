import Foundation
import MLX
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

internal enum Qwen35LayerKind: String, Codable, Sendable, Equatable {
    case linearAttention = "linear_attention"
    case fullAttention = "full_attention"

    internal static func defaultSchedule(
        layerCount: Int,
        fullAttentionInterval: Int
    ) -> [Qwen35LayerKind] {
        precondition(fullAttentionInterval > 0, "full_attention_interval must be positive")
        return (0 ..< layerCount).map { layerIndex in
            (layerIndex + 1) % fullAttentionInterval == 0
                ? .fullAttention
                : .linearAttention
        }
    }
}

internal struct Qwen35LayerSchedule: Sendable, Equatable {
    internal let kinds: [Qwen35LayerKind]
    internal let firstLinearIndex: Int?
    internal let firstFullAttentionIndex: Int?

    internal init(kinds: [Qwen35LayerKind]) {
        self.kinds = kinds
        self.firstLinearIndex = kinds.firstIndex(of: .linearAttention)
        self.firstFullAttentionIndex = kinds.firstIndex(of: .fullAttention)
    }

    internal func kind(at layerIndex: Int) -> Qwen35LayerKind {
        kinds[layerIndex]
    }
}

internal struct Qwen35AttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let queryProjectionDimensions: Int
    internal let keyValueProjectionDimensions: Int
    internal let outputProjectionDimensions: Int
    internal let rotaryDimensions: Int
    internal let scale: Float

    internal init(_ configuration: Qwen35TextConfiguration) {
        let headDimensions =
            configuration.headDim ?? configuration.hiddenSize / configuration.attentionHeads
        precondition(configuration.hiddenSize > 0, "hidden_size must be positive")
        precondition(configuration.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(configuration.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(headDimensions > 0, "head_dim must be positive")

        self.hiddenSize = configuration.hiddenSize
        self.attentionHeads = configuration.attentionHeads
        self.keyValueHeads = configuration.kvHeads
        self.headDimensions = headDimensions
        self.queryProjectionDimensions = configuration.attentionHeads * headDimensions
        self.keyValueProjectionDimensions = configuration.kvHeads * headDimensions
        self.outputProjectionDimensions = configuration.attentionHeads * headDimensions
        self.rotaryDimensions = max(
            1,
            Int(Float(headDimensions) * configuration.partialRotaryFactor)
        )
        self.scale = pow(Float(headDimensions), -0.5)
    }
}

internal struct Qwen35LinearAttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let valueHeads: Int
    internal let keyHeads: Int
    internal let keyHeadDimensions: Int
    internal let valueHeadDimensions: Int
    internal let keyDimensions: Int
    internal let valueDimensions: Int
    internal let convolutionKernelSize: Int
    internal let convolutionDimensions: Int

    internal init(_ configuration: Qwen35TextConfiguration) {
        precondition(configuration.linearNumValueHeads > 0, "linear_num_value_heads must be positive")
        precondition(configuration.linearNumKeyHeads > 0, "linear_num_key_heads must be positive")
        precondition(
            configuration.linearNumValueHeads % configuration.linearNumKeyHeads == 0,
            "linear_num_value_heads must be divisible by linear_num_key_heads"
        )
        precondition(configuration.linearKeyHeadDim > 0, "linear_key_head_dim must be positive")
        precondition(configuration.linearValueHeadDim > 0, "linear_value_head_dim must be positive")
        precondition(configuration.linearConvKernelDim > 1, "linear_conv_kernel_dim must exceed one")

        self.hiddenSize = configuration.hiddenSize
        self.valueHeads = configuration.linearNumValueHeads
        self.keyHeads = configuration.linearNumKeyHeads
        self.keyHeadDimensions = configuration.linearKeyHeadDim
        self.valueHeadDimensions = configuration.linearValueHeadDim
        self.keyDimensions = keyHeads * keyHeadDimensions
        self.valueDimensions = valueHeads * valueHeadDimensions
        self.convolutionKernelSize = configuration.linearConvKernelDim
        self.convolutionDimensions = keyDimensions * 2 + valueDimensions
    }
}

internal struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4
    var layerTypes: [Qwen35LayerKind] = []
    var mlpOnlyLayers: [Int] = []
    var mtpNumHiddenLayers: Int = 0

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case layerTypes = "layer_types"
        case mlpOnlyLayers = "mlp_only_layers"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        self.mlpOnlyLayers =
            try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }

        if let layerTypeNames = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            guard layerTypeNames.count == self.hiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes,
                    in: container,
                    debugDescription:
                        "layer_types count must match num_hidden_layers"
                )
            }
            self.layerTypes = try layerTypeNames.map { name in
                guard let kind = Qwen35LayerKind(rawValue: name) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .layerTypes,
                        in: container,
                        debugDescription: "Unsupported Qwen3.5 layer type: \(name)"
                    )
                }
                return kind
            }
        } else {
            self.layerTypes = Qwen35LayerKind.defaultSchedule(
                layerCount: self.hiddenLayers,
                fullAttentionInterval: self.fullAttentionInterval
            )
        }
    }

    internal var hasNativeMTP: Bool {
        mtpNumHiddenLayers > 0
    }

    internal var layerSchedule: Qwen35LayerSchedule {
        Qwen35LayerSchedule(kinds: layerTypes)
    }

    internal func layerKind(at layerIndex: Int) -> Qwen35LayerKind {
        layerTypes[layerIndex]
    }

    internal func usesMoE(layerIndex: Int) -> Bool {
        numExperts > 0
            && numExpertsPerTok > 0
            && !mlpOnlyLayers.contains(layerIndex)
            && (layerIndex + 1) % decoderSparseStep == 0
    }
}

private protocol Qwen35NativeMTPActivationStrategy {
    func shouldActivateNativeMTP(
        configuration: Qwen35TextConfiguration,
        weights: [String: MLXArray]
    ) -> Bool

    func removingNativeMTPWeights(from weights: [String: MLXArray]) -> [String: MLXArray]
}

private struct Qwen35WeightBackedNativeMTPActivationStrategy:
    Qwen35NativeMTPActivationStrategy
{
    private let requiredProjectionKeys = [
        "mtp.fc.weight",
        "language_model.mtp.fc.weight",
    ]

    func shouldActivateNativeMTP(
        configuration: Qwen35TextConfiguration,
        weights: [String: MLXArray]
    ) -> Bool {
        guard configuration.hasNativeMTP else {
            return false
        }
        return requiredProjectionKeys.contains { weights[$0] != nil }
    }

    func removingNativeMTPWeights(from weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !isNativeMTPWeightKey($0.key) }
    }

    private func isNativeMTPWeightKey(_ key: String) -> Bool {
        key.hasPrefix("mtp.") ||
            key.hasPrefix("language_model.mtp.")
    }
}

// MARK: - Linear Attention

final class Qwen35LinearAttention: Module {
    let layout: Qwen35LinearAttentionLayout

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.layout = Qwen35LinearAttentionLayout(args)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: layout.convolutionDimensions,
            outputChannels: layout.convolutionDimensions,
            kernelSize: layout.convolutionKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: layout.convolutionDimensions,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(
            layout.hiddenSize,
            layout.convolutionDimensions,
            bias: false
        )
        _inProjZ.wrappedValue = Linear(layout.hiddenSize, layout.valueDimensions, bias: false)
        _inProjB.wrappedValue = Linear(layout.hiddenSize, layout.valueHeads, bias: false)
        _inProjA.wrappedValue = Linear(layout.hiddenSize, layout.valueHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([layout.valueHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [layout.valueHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(
            dimensions: layout.valueHeadDimensions,
            eps: args.rmsNormEps
        )
        _outProj.wrappedValue = Linear(
            layout.valueDimensions,
            layout.hiddenSize,
            bias: false
        )

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(
            B,
            S,
            layout.valueHeads,
            layout.valueHeadDimensions
        )
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros(
                [B, layout.convolutionKernelSize - 1, layout.convolutionDimensions],
                dtype: inputs.dtype
            )
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (1 - layout.convolutionKernelSize)...]
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(
            convOut,
            indices: [layout.keyDimensions, 2 * layout.keyDimensions],
            axis: -1
        )
        let q = convSplit[0].reshaped(
            B,
            S,
            layout.keyHeads,
            layout.keyHeadDimensions
        )
        let k = convSplit[1].reshaped(
            B,
            S,
            layout.keyHeads,
            layout.keyHeadDimensions
        )
        let v = convSplit[2].reshaped(
            B,
            S,
            layout.valueHeads,
            layout.valueHeadDimensions
        )

        var state = cache?[1]
        let dtype = q.dtype
        let invScale = pow(Float(layout.keyHeadDimensions), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        var out: MLXArray

        (out, state) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask
        )

        if let cache {
            cache[1] = state
        }

        out = norm(out, gate: z)
        return outProj(out.reshaped(B, S, -1))
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let layout: Qwen35AttentionLayout

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextConfiguration) {
        self.layout = Qwen35AttentionLayout(args)

        _qProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionDimensions * 2,
            bias: args.attentionBias
        )
        _kProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionDimensions,
            bias: args.attentionBias
        )
        _vProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionDimensions,
            bias: args.attentionBias
        )
        _oProj.wrappedValue = Linear(
            layout.outputProjectionDimensions,
            layout.hiddenSize,
            bias: args.attentionBias
        )

        _qNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: args.rmsNormEps)

        self.rope = initializeRope(
            dims: layout.rotaryDimensions,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(
            B,
            L,
            layout.attentionHeads,
            -1
        )
        .split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, layout.keyValueHeads, -1)).transposed(0, 2, 1, 3)
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

        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let layerKind: Qwen35LayerKind

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35LinearAttention?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.layerKind = args.layerKind(at: layerIdx)

        if layerKind == .linearAttention {
            _linearAttn.wrappedValue = Qwen35LinearAttention(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.usesMoE(layerIndex: layerIdx) {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    var isLinear: Bool {
        layerKind == .linearAttention
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            guard let linearAttn else {
                preconditionFailure("Qwen3.5 linear layer is missing its linear attention module")
            }
            r = linearAttn(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            guard let selfAttn else {
                preconditionFailure("Qwen3.5 full-attention layer is missing its attention module")
            }
            r = selfAttn(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

// MARK: - MTP

final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        if args.numExperts > 0 && args.numExpertsPerTok > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: attentionMask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear

    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in Qwen35MTPDecoderLayer(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        nextTokenIDs: MLXArray,
        embedTokens: Embedding,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var cacheArray = cache?.map { Optional($0) }
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let tokenEmbeddings = embedTokens(nextTokenIDs)
        var h = concatenated(
            [
                preFCNormEmbedding(tokenEmbeddings),
                preFCNormHidden(hiddenStates)
            ],
            axis: -1
        )
        h = fc(h)

        let firstCache: KVCache?
        if let cacheArray {
            firstCache = cacheArray.first ?? nil
        } else {
            firstCache = nil
        }
        let attentionMask = createAttentionMask(h: h, cache: firstCache)
        for (index, layer) in layers.enumerated() {
            let layerCache = cacheArray?[index] ?? nil
            h = layer(h, attentionMask: attentionMask, cache: layerCache)
        }
        return norm(h)
    }
}

// MARK: - Text Model

internal class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm
    let schedule: Qwen35LayerSchedule

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)
        self.schedule = args.layerSchedule

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    func hiddenStates(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let fullAttentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let index = schedule.firstFullAttentionIndex {
            fullAttentionMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[index])
        } else {
            fullAttentionMask = .none
        }

        let ssmMask: MLXArray?
        if let index = schedule.firstLinearIndex {
            ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[index] as? MambaCache)
        } else {
            ssmMask = nil
        }

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : fullAttentionMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
        }

        return hiddenStates
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        norm(hiddenStates(inputs, cache: cache))
    }
}

internal class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider, NativeMTPModel,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") fileprivate var model: Qwen35TextModelInner

    let configuration: Qwen35TextConfiguration
    private let nativeMTPActivationStrategy: Qwen35NativeMTPActivationStrategy

    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPModule?

    internal init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self._model.wrappedValue = Qwen35TextModelInner(args)
        self.nativeMTPActivationStrategy = Qwen35WeightBackedNativeMTPActivationStrategy()

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        if args.hasNativeMTP {
            _mtp.wrappedValue = Qwen35MTPModule(args)
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectToLogits(model(inputs, cache: cache))
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { layer in
            if layer.isLinear {
                MambaCache()
            } else {
                KVCacheSimple()
            }
        }
    }

    internal var hasNativeMTP: Bool {
        mtp != nil
    }

    internal var supportsNativeMTP: Bool {
        hasNativeMTP
    }

    internal func makeMTPCache(parameters: GenerateParameters?) -> [KVCache] {
        Array(repeating: KVCacheSimple(), count: mtp?.layers.count ?? 0)
    }

    internal func mtpForward(
        hiddenStates: MLXArray,
        nextTokenIDs: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray? {
        guard let mtp else { return nil }
        return projectToLogits(mtpHiddenStates(
            mtp: mtp,
            hiddenStates: hiddenStates,
            nextTokenIDs: nextTokenIDs,
            cache: cache
        ))
    }

    internal func nativeMTPMainOutput(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> NativeMTPMainOutput {
        let hiddenStates = model.hiddenStates(input.tokens, cache: cache)
        return NativeMTPMainOutput(
            logits: projectToLogits(model.norm(hiddenStates)),
            hiddenStates: hiddenStates,
            state: state
        )
    }

    internal func nativeMTPDraftOutput(
        hiddenStates: MLXArray,
        nextTokenIDs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPDraftOutput? {
        guard let mtp else { return nil }
        let draftHiddenStates = mtpHiddenStates(
            mtp: mtp,
            hiddenStates: hiddenStates,
            nextTokenIDs: nextTokenIDs,
            cache: cache
        )
        return NativeMTPDraftOutput(
            logits: projectToLogits(draftHiddenStates),
            hiddenStates: draftHiddenStates
        )
    }

    internal func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.hiddenStates(inputs, cache: cache)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        greedyTokenOutput(
            logits: projectToLogits(
                lastTokenHiddenState(
                    model(input[text: .newAxis].tokens, cache: cache)
                )
            ),
            state: state
        )
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = normalizedMTPWeights(weights)
        let hasMTPWeights = weights.keys.contains { $0.contains("mtp.") }
        let shouldActivateNativeMTP = nativeMTPActivationStrategy.shouldActivateNativeMTP(
            configuration: configuration,
            weights: weights
        )
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasMTPWeights || hasUnsanitizedConv1d

        if !shouldActivateNativeMTP {
            disableNativeMTPModule()
            weights = nativeMTPActivationStrategy.removingNativeMTPWeights(from: weights)
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["language_model.lm_head.weight"] = nil
        }

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "language_model.mtp.norm.weight",
            "language_model.mtp.pre_fc_norm_hidden.weight",
            "language_model.mtp.pre_fc_norm_embedding.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }

    private func projectToLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }

    private func disableNativeMTPModule() {
        guard mtp != nil else {
            return
        }
        var modules = ModuleChildren()
        modules["mtp"] = NestedItem<String, Module>.none
        update(modules: modules)
    }

    private func mtpHiddenStates(
        mtp: Qwen35MTPModule,
        hiddenStates: MLXArray,
        nextTokenIDs: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        mtp(
            hiddenStates: hiddenStates,
            nextTokenIDs: nextTokenIDs,
            embedTokens: model.embedTokens,
            cache: cache
        )
    }

    private func normalizedMTPWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var normalized = [String: MLXArray]()
        normalized.reserveCapacity(weights.count)

        for (key, value) in weights {
            normalized[normalizedMTPKey(key)] = value
        }
        return normalized
    }

    private func normalizedMTPKey(_ key: String) -> String {
        if key.hasPrefix("model.language_model.mtp.") {
            return "mtp." + key.dropFirst("model.language_model.mtp.".count)
        }
        if key.hasPrefix("model.mtp.") {
            return "mtp." + key.dropFirst("model.mtp.".count)
        }
        return key
    }
}

extension Qwen35TextModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }
}

internal enum Qwen35TopLevelWeightMapper {
    internal static func languageModelWeights(from weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped = [String: MLXArray]()
        mapped.reserveCapacity(weights.count)

        for (originalKey, value) in weights {
            guard !shouldDropTopLevelWeight(originalKey) else {
                continue
            }
            mapped[languageModelKey(for: originalKey)] = value
        }

        return mapped
    }

    private static func shouldDropTopLevelWeight(_ key: String) -> Bool {
        key.hasPrefix("vision_tower") || key.hasPrefix("model.visual")
    }

    private static func languageModelKey(for key: String) -> String {
        if key.hasPrefix("model.language_model.mtp.") {
            return "language_model.mtp."
                + key.dropFirst("model.language_model.mtp.".count)
        }
        if key.hasPrefix("model.mtp.") {
            return "language_model.mtp." + key.dropFirst("model.mtp.".count)
        }
        if key.hasPrefix("model.language_model") {
            return key.replacingOccurrences(
                of: "model.language_model",
                with: "language_model.model"
            )
        }
        if key.hasPrefix("language_model.") {
            return key
        }
        return "language_model." + key
    }
}

// MARK: - Top-level Model

internal class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider, NativeMTPModel,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    internal init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        languageModel.greedyToken(input, cache: cache, state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    internal var supportsNativeMTP: Bool {
        languageModel.supportsNativeMTP
    }

    internal func makeMTPCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeMTPCache(parameters: parameters)
    }

    internal func nativeMTPMainOutput(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> NativeMTPMainOutput {
        languageModel.nativeMTPMainOutput(input, cache: cache, state: state)
    }

    internal func nativeMTPDraftOutput(
        hiddenStates: MLXArray,
        nextTokenIDs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPDraftOutput? {
        languageModel.nativeMTPDraftOutput(
            hiddenStates: hiddenStates,
            nextTokenIDs: nextTokenIDs,
            cache: cache
        )
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        languageModel.sanitize(
            weights: Qwen35TopLevelWeightMapper.languageModelWeights(from: weights)
        )
    }
}

extension Qwen35Model: LoRAModel {
    internal var loraLayers: [Module] {
        languageModel.model.layers
    }
}
