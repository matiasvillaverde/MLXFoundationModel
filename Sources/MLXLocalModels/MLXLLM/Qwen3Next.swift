import Foundation
import MLX
import MLXNN

// MARK: - Shared Helpers

func sigmoidMultiply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
    x * sigmoid(gate)
}

final class Qwen3NextRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        let normalized = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        guard let gate else {
            return normalized
        }
        return normalized * silu(gate)
    }
}

final class Qwen3NextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Configuration

internal enum Qwen3NextLayerKind: String, Codable, Sendable, Equatable {
    case linearAttention = "linear_attention"
    case fullAttention = "full_attention"

    internal static func defaultSchedule(
        layerCount: Int,
        fullAttentionInterval: Int
    ) -> [Qwen3NextLayerKind] {
        precondition(layerCount > 0, "Qwen3Next layer count must be positive")
        precondition(
            fullAttentionInterval > 0,
            "Qwen3Next full_attention_interval must be positive"
        )
        return (0 ..< layerCount).map { layerIndex in
            (layerIndex + 1).isMultiple(of: fullAttentionInterval)
                ? .fullAttention
                : .linearAttention
        }
    }
}

internal struct Qwen3NextConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var decoderSparseStep: Int
    var sharedExpertIntermediateSize: Int
    var mlpOnlyLayers: [Int]
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var maxPositionEmbeddings: Int
    var normTopkProb: Bool
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int
    var layerTypes: [Qwen3NextLayerKind]

    internal init(
        modelType: String = "qwen3_next",
        hiddenSize: Int = 2_048,
        hiddenLayers: Int = 32,
        intermediateSize: Int = 6_144,
        attentionHeads: Int = 32,
        linearNumValueHeads: Int = 32,
        linearNumKeyHeads: Int = 4,
        linearKeyHeadDim: Int = 64,
        linearValueHeadDim: Int = 64,
        linearConvKernelDim: Int = 4,
        numExperts: Int = 0,
        numExpertsPerTok: Int = 0,
        decoderSparseStep: Int = 1,
        sharedExpertIntermediateSize: Int = 0,
        mlpOnlyLayers: [Int] = [],
        moeIntermediateSize: Int = 0,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 151_936,
        kvHeads: Int = 4,
        ropeTheta: Float = 1_000_000,
        partialRotaryFactor: Float = 1.0,
        maxPositionEmbeddings: Int = 32_768,
        normTopkProb: Bool = false,
        tieWordEmbeddings: Bool = false,
        attentionBias: Bool = false,
        headDim: Int? = nil,
        ropeScaling: [String: StringOrNumber]? = nil,
        fullAttentionInterval: Int = 4,
        layerTypes: [Qwen3NextLayerKind]? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.linearNumValueHeads = linearNumValueHeads
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelDim = linearConvKernelDim
        self.numExperts = numExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.decoderSparseStep = decoderSparseStep
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.mlpOnlyLayers = mlpOnlyLayers
        self.moeIntermediateSize = moeIntermediateSize
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normTopkProb = normTopkProb
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.headDim = headDim
        self.ropeScaling = ropeScaling
        self.fullAttentionInterval = fullAttentionInterval
        self.layerTypes = layerTypes ?? Qwen3NextLayerKind.defaultSchedule(
            layerCount: hiddenLayers,
            fullAttentionInterval: fullAttentionInterval
        )
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case layerTypes = "layer_types"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        let fullAttentionInterval = try container.decodeIfPresent(
            Int.self,
            forKey: .fullAttentionInterval
        ) ?? 4
        let layerTypes = try Self.decodeLayerTypes(
            from: container,
            hiddenLayers: hiddenLayers,
            fullAttentionInterval: fullAttentionInterval
        )

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "qwen3_next",
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048,
            hiddenLayers: hiddenLayers,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? 6_144,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? 32,
            linearNumValueHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .linearNumValueHeads
            ) ?? 32,
            linearNumKeyHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .linearNumKeyHeads
            ) ?? 4,
            linearKeyHeadDim: try container.decodeIfPresent(
                Int.self,
                forKey: .linearKeyHeadDim
            ) ?? 64,
            linearValueHeadDim: try container.decodeIfPresent(
                Int.self,
                forKey: .linearValueHeadDim
            ) ?? 64,
            linearConvKernelDim: try container.decodeIfPresent(
                Int.self,
                forKey: .linearConvKernelDim
            ) ?? 4,
            numExperts: try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0,
            numExpertsPerTok: try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
                ?? 0,
            decoderSparseStep: try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep)
                ?? 1,
            sharedExpertIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .sharedExpertIntermediateSize
            ) ?? 0,
            mlpOnlyLayers: try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers)
                ?? [],
            moeIntermediateSize: try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
                ?? 0,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 151_936,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000,
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .partialRotaryFactor
            ) ?? 1.0,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 32_768,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            fullAttentionInterval: fullAttentionInterval,
            layerTypes: layerTypes
        )
    }

    private static func decodeLayerTypes(
        from container: KeyedDecodingContainer<CodingKeys>,
        hiddenLayers: Int,
        fullAttentionInterval: Int
    ) throws -> [Qwen3NextLayerKind] {
        guard let names = try container.decodeIfPresent([String].self, forKey: .layerTypes) else {
            return Qwen3NextLayerKind.defaultSchedule(
                layerCount: hiddenLayers,
                fullAttentionInterval: fullAttentionInterval
            )
        }
        guard names.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription: "layer_types count must match num_hidden_layers"
            )
        }
        return try names.map { name in
            guard let kind = Qwen3NextLayerKind(rawValue: name) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes,
                    in: container,
                    debugDescription: "Unsupported Qwen3Next layer type: \(name)"
                )
            }
            return kind
        }
    }

    internal func usesMoE(layerIndex: Int) -> Bool {
        numExperts > 0
            && numExpertsPerTok > 0
            && !mlpOnlyLayers.contains(layerIndex)
            && (layerIndex + 1).isMultiple(of: decoderSparseStep)
    }
}

// MARK: - Plans

internal struct Qwen3NextLayerPlan: Sendable, Equatable {
    internal let kinds: [Qwen3NextLayerKind]
    internal let firstLinearIndex: Int?
    internal let firstFullAttentionIndex: Int?

    internal init(_ configuration: Qwen3NextConfiguration) {
        precondition(configuration.hiddenLayers > 0, "Qwen3Next must have at least one layer")
        precondition(
            configuration.layerTypes.count == configuration.hiddenLayers,
            "Qwen3Next layer_types count must match hidden layer count"
        )
        self.kinds = configuration.layerTypes
        self.firstLinearIndex = kinds.firstIndex(of: .linearAttention)
        self.firstFullAttentionIndex = kinds.firstIndex(of: .fullAttention)
    }

    internal var count: Int {
        kinds.count
    }

    internal func kind(at layerIndex: Int) -> Qwen3NextLayerKind {
        kinds[layerIndex]
    }

    internal func isLinear(layerIndex: Int) -> Bool {
        kind(at: layerIndex) == .linearAttention
    }
}

internal struct Qwen3NextAttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let queryProjectionDimensions: Int
    internal let keyValueProjectionDimensions: Int
    internal let rotaryDimensions: Int
    internal let scale: Float

    internal init(_ configuration: Qwen3NextConfiguration) {
        let headDimensions =
            configuration.headDim ?? configuration.hiddenSize / configuration.attentionHeads

        precondition(configuration.hiddenSize > 0, "Qwen3Next hidden size must be positive")
        precondition(
            configuration.attentionHeads > 0,
            "Qwen3Next attention head count must be positive"
        )
        precondition(configuration.kvHeads > 0, "Qwen3Next KV head count must be positive")
        precondition(headDimensions > 0, "Qwen3Next head dimension must be positive")
        precondition(
            configuration.attentionHeads.isMultiple(of: configuration.kvHeads),
            "Qwen3Next attention heads must group KV heads"
        )
        precondition(
            configuration.partialRotaryFactor > 0,
            "Qwen3Next partial rotary factor must be positive"
        )

        self.hiddenSize = configuration.hiddenSize
        self.attentionHeads = configuration.attentionHeads
        self.keyValueHeads = configuration.kvHeads
        self.headDimensions = headDimensions
        self.queryProjectionDimensions = configuration.attentionHeads * headDimensions
        self.keyValueProjectionDimensions = configuration.kvHeads * headDimensions
        self.rotaryDimensions = max(
            1,
            Int(Float(headDimensions) * configuration.partialRotaryFactor)
        )
        self.scale = pow(Float(headDimensions), -0.5)
    }
}

internal struct Qwen3NextLinearAttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let valueHeads: Int
    internal let keyHeads: Int
    internal let keyHeadDimensions: Int
    internal let valueHeadDimensions: Int
    internal let keyDimensions: Int
    internal let valueDimensions: Int
    internal let convolutionKernelSize: Int
    internal let convolutionDimensions: Int
    internal let valueHeadsPerKeyHead: Int

    internal init(_ configuration: Qwen3NextConfiguration) {
        precondition(configuration.hiddenSize > 0, "Qwen3Next hidden size must be positive")
        precondition(
            configuration.linearNumValueHeads > 0,
            "Qwen3Next linear value heads must be positive"
        )
        precondition(
            configuration.linearNumKeyHeads > 0,
            "Qwen3Next linear key heads must be positive"
        )
        precondition(
            configuration.linearNumValueHeads.isMultiple(of: configuration.linearNumKeyHeads),
            "Qwen3Next linear value heads must divide evenly by key heads"
        )
        precondition(
            configuration.linearKeyHeadDim > 0,
            "Qwen3Next linear key head dimension must be positive"
        )
        precondition(
            configuration.linearValueHeadDim > 0,
            "Qwen3Next linear value head dimension must be positive"
        )
        precondition(
            configuration.linearConvKernelDim > 1,
            "Qwen3Next convolution kernel size must exceed one"
        )

        self.hiddenSize = configuration.hiddenSize
        self.valueHeads = configuration.linearNumValueHeads
        self.keyHeads = configuration.linearNumKeyHeads
        self.keyHeadDimensions = configuration.linearKeyHeadDim
        self.valueHeadDimensions = configuration.linearValueHeadDim
        self.keyDimensions = keyHeads * keyHeadDimensions
        self.valueDimensions = valueHeads * valueHeadDimensions
        self.convolutionKernelSize = configuration.linearConvKernelDim
        self.convolutionDimensions = keyDimensions * 2 + valueDimensions
        self.valueHeadsPerKeyHead = valueHeads / keyHeads
    }
}

internal struct Qwen3NextMoEPlan: Sendable, Equatable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let normalizesProbabilities: Bool

    internal init(_ configuration: Qwen3NextConfiguration) {
        precondition(configuration.numExperts > 0, "Qwen3Next MoE expert count must be positive")
        precondition(configuration.numExpertsPerTok > 0, "Qwen3Next MoE top-k must be positive")
        precondition(
            configuration.numExpertsPerTok <= configuration.numExperts,
            "Qwen3Next MoE top-k cannot exceed expert count"
        )
        precondition(
            configuration.moeIntermediateSize > 0,
            "Qwen3Next MoE intermediate size must be positive"
        )
        precondition(
            configuration.sharedExpertIntermediateSize > 0,
            "Qwen3Next shared expert intermediate size must be positive"
        )

        self.expertCount = configuration.numExperts
        self.selectedExpertCount = configuration.numExpertsPerTok
        self.normalizesProbabilities = configuration.normTopkProb
    }

    internal func route(_ logits: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        let probabilities = MLX.softmax(logits, axis: -1, precise: true)
        let partitionStart = expertCount - selectedExpertCount
        let indices = MLX.argPartition(
            probabilities,
            kth: partitionStart,
            axis: -1
        )[.ellipsis, partitionStart...]
        var scores = MLX.takeAlong(probabilities, indices, axis: -1)
        if normalizesProbabilities {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (scores, indices)
    }
}

private enum Qwen3NextExpertProjection: String, CaseIterable {
    case up = "up_proj"
    case down = "down_proj"
    case gate = "gate_proj"
}

internal struct Qwen3NextExpertPackingPlan: Sendable {
    internal let configuration: Qwen3NextConfiguration
    internal let layerPlan: Qwen3NextLayerPlan

    internal init(_ configuration: Qwen3NextConfiguration) {
        self.configuration = configuration
        self.layerPlan = Qwen3NextLayerPlan(configuration)
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        guard configuration.numExperts > 0 else {
            return weights
        }

        var packed = weights
        for layerIndex in 0 ..< layerPlan.count where configuration.usesMoE(layerIndex: layerIndex) {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in Qwen3NextExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let keys = (0 ..< configuration.numExperts).map { expertIndex in
                        "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                    }
                    guard keys.allSatisfy({ packed[$0] != nil }) else {
                        continue
                    }
                    let tensors = keys.map { packed.removeValue(forKey: $0)! }
                    packed["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        MLX.stacked(tensors)
                }
            }
        }
        return packed
    }
}

internal struct Qwen3NextSanitizerPlan: Sendable {
    internal let configuration: Qwen3NextConfiguration

    internal init(_ configuration: Qwen3NextConfiguration) {
        self.configuration = configuration
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        sanitized = sanitized.filter { key, _ in
            !key.contains("mtp.") && !key.contains("rotary_emb.inv_freq")
        }
        sanitized = Qwen3NextExpertPackingPlan(configuration).pack(sanitized)

        for key in Array(sanitized.keys) {
            guard let value = sanitized[key] else { continue }
            if key.contains("conv1d.weight"), value.dim(-1) != 1 {
                sanitized[key] = value.movedAxis(source: 2, destination: 1)
                continue
            }
            if Self.normWeightSuffixes.contains(where: { key.hasSuffix($0) }), value.ndim == 1 {
                sanitized[key] = value + MLXArray(1, dtype: value.dtype)
            }
        }

        return sanitized
    }

    private static let normWeightSuffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ]
}

// MARK: - Model Components

internal final class Qwen3NextAttention: Module {
    let layout: Qwen3NextAttentionLayout

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ configuration: Qwen3NextConfiguration) {
        self.layout = Qwen3NextAttentionLayout(configuration)

        _qProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionDimensions * 2,
            bias: configuration.attentionBias
        )
        _kProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionDimensions,
            bias: configuration.attentionBias
        )
        _vProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionDimensions,
            bias: configuration.attentionBias
        )
        _oProj.wrappedValue = Linear(
            layout.queryProjectionDimensions,
            layout.hiddenSize,
            bias: configuration.attentionBias
        )
        _qNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: configuration.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: layout.headDimensions, eps: configuration.rmsNormEps)
        self.rope = initializeRope(
            dims: layout.rotaryDimensions,
            base: configuration.ropeTheta,
            traditional: false,
            scalingConfig: configuration.ropeScaling,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings
        )
        super.init()
    }

    internal func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let tokenCount = x.dim(1)

        let qProjected = qProj(x)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, -1)
            .split(parts: 2, axis: -1)

        var queries = qNorm(qProjected[0]).transposed(0, 2, 1, 3)
        let gate = qProjected[1].reshaped(batchSize, tokenCount, -1)
        var keys = kNorm(
            kProj(x).reshaped(batchSize, tokenCount, layout.keyValueHeads, -1)
        )
        .transposed(0, 2, 1, 3)
        let values = vProj(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, -1)
            .transposed(0, 2, 1, 3)

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
        .reshaped(batchSize, tokenCount, -1)

        return oProj(sigmoidMultiply(output, gate))
    }
}

internal final class Qwen3NextGatedDeltaNet: Module {
    let layout: Qwen3NextLinearAttentionLayout

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkvz") var inProjQKVZ: Linear
    @ModuleInfo(key: "in_proj_ba") var inProjBA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ configuration: Qwen3NextConfiguration) {
        self.layout = Qwen3NextLinearAttentionLayout(configuration)

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
        _inProjQKVZ.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyDimensions * 2 + layout.valueDimensions * 2,
            bias: false
        )
        _inProjBA.wrappedValue = Linear(layout.hiddenSize, layout.valueHeads * 2, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([layout.valueHeads])
        _aLog.wrappedValue = log(MLXRandom.uniform(low: 0, high: 16, [layout.valueHeads]))
        _norm.wrappedValue = Qwen3NextRMSNormGated(
            dimensions: layout.valueHeadDimensions,
            eps: configuration.rmsNormEps
        )
        _outProj.wrappedValue = Linear(
            layout.valueDimensions,
            layout.hiddenSize,
            bias: false
        )
        super.init()
    }

    private func splitProjectedStates(
        qkvz: MLXArray,
        ba: MLXArray
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray) {
        let batchSize = qkvz.dim(0)
        let tokenCount = qkvz.dim(1)

        let qkvzByKeyHead = qkvz.reshaped(batchSize, tokenCount, layout.keyHeads, -1)
        let qkvzParts = MLX.split(
            qkvzByKeyHead,
            indices: [
                layout.keyHeadDimensions,
                layout.keyHeadDimensions * 2,
                layout.keyHeadDimensions * 2
                    + layout.valueHeadsPerKeyHead * layout.valueHeadDimensions,
            ],
            axis: -1
        )

        let baByKeyHead = ba.reshaped(batchSize, tokenCount, layout.keyHeads, -1)
        let baParts = MLX.split(baByKeyHead, indices: [layout.valueHeadsPerKeyHead], axis: -1)

        return (
            q: qkvzParts[0],
            k: qkvzParts[1],
            v: qkvzParts[2].reshaped(batchSize, tokenCount, -1, layout.valueHeadDimensions),
            z: qkvzParts[3].reshaped(batchSize, tokenCount, -1, layout.valueHeadDimensions),
            b: baParts[0].reshaped(batchSize, tokenCount, layout.valueHeads),
            a: baParts[1].reshaped(batchSize, tokenCount, layout.valueHeads)
        )
    }

    internal func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil
    ) -> MLXArray {
        let batchSize = inputs.dim(0)
        let tokenCount = inputs.dim(1)
        let projected = splitProjectedStates(qkvz: inProjQKVZ(inputs), ba: inProjBA(inputs))

        let convState = cache?[0] ?? MLXArray.zeros(
            [batchSize, layout.convolutionKernelSize - 1, layout.convolutionDimensions],
            dtype: inputs.dtype
        )

        var convTokens = concatenated(
            [
                projected.q.reshaped(batchSize, tokenCount, -1),
                projected.k.reshaped(batchSize, tokenCount, -1),
                projected.v.reshaped(batchSize, tokenCount, -1),
            ],
            axis: -1
        )

        if let mask {
            convTokens = MLX.where(
                expandedDimensions(mask, axis: -1),
                convTokens,
                MLXArray.zeros(like: convTokens)
            )
        }

        let convInput = concatenated([convState, convTokens], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (1 - layout.convolutionKernelSize)..., 0...]
        }

        let convOutput = silu(conv1d(convInput))
        let convParts = MLX.split(
            convOutput,
            indices: [layout.keyDimensions, layout.keyDimensions * 2],
            axis: -1
        )

        var q = convParts[0].reshaped(
            batchSize,
            tokenCount,
            layout.keyHeads,
            layout.keyHeadDimensions
        )
        var k = convParts[1].reshaped(
            batchSize,
            tokenCount,
            layout.keyHeads,
            layout.keyHeadDimensions
        )
        let v = convParts[2].reshaped(
            batchSize,
            tokenCount,
            layout.valueHeads,
            layout.valueHeadDimensions
        )

        let inverseScale = pow(Float(layout.keyHeadDimensions), -0.5)
        q = MLXArray(inverseScale * inverseScale).asType(inputs.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k = MLXArray(inverseScale).asType(inputs.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (output, state) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: projected.a,
            b: projected.b,
            aLog: aLog,
            dtBias: dtBias,
            state: cache?[1],
            mask: mask
        )

        if let cache {
            cache[1] = state
        }

        return outProj(norm(output, gate: projected.z).reshaped(batchSize, tokenCount, -1))
    }
}

internal final class Qwen3NextSparseMoeBlock: Module, UnaryLayer {
    let plan: Qwen3NextMoEPlan

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ configuration: Qwen3NextConfiguration) {
        self.plan = Qwen3NextMoEPlan(configuration)

        _gate.wrappedValue = Linear(configuration.hiddenSize, configuration.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.moeIntermediateSize,
            numExperts: configuration.numExperts
        )
        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(configuration.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let route = plan.route(gate(x))
        let routed = switchMLP(x, route.indices)
        let combined = (routed * route.scores[.ellipsis, .newAxis]).sum(axis: -2)
        let shared = sigmoid(sharedExpertGate(x)) * sharedExpert(x)
        return combined + shared
    }
}

internal final class Qwen3NextDecoderLayer: Module {
    let layerKind: Qwen3NextLayerKind

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3NextAttention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen3NextGatedDeltaNet?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var feedForward: Module & UnaryLayer

    init(
        _ configuration: Qwen3NextConfiguration,
        layerIndex: Int,
        layerPlan: Qwen3NextLayerPlan
    ) {
        self.layerKind = layerPlan.kind(at: layerIndex)

        switch layerKind {
        case .linearAttention:
            _linearAttn.wrappedValue = Qwen3NextGatedDeltaNet(configuration)
        case .fullAttention:
            _selfAttn.wrappedValue = Qwen3NextAttention(configuration)
        }

        if configuration.usesMoE(layerIndex: layerIndex) {
            _feedForward.wrappedValue = Qwen3NextSparseMoeBlock(configuration)
        } else {
            _feedForward.wrappedValue = Qwen3NextMLP(
                dimensions: configuration.hiddenSize,
                hiddenDimensions: configuration.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
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
        let residual: MLXArray
        switch layerKind {
        case .linearAttention:
            guard let linearAttn else {
                preconditionFailure("Qwen3Next linear layer is missing linear attention")
            }
            residual = linearAttn(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        case .fullAttention:
            guard let selfAttn else {
                preconditionFailure("Qwen3Next full-attention layer is missing attention")
            }
            residual = selfAttn(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let hidden = x + residual
        return hidden + feedForward(postAttentionLayerNorm(hidden))
    }
}

internal final class Qwen3NextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen3NextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerPlan: Qwen3NextLayerPlan

    init(_ configuration: Qwen3NextConfiguration) {
        precondition(configuration.vocabularySize > 0, "Qwen3Next vocabulary size must be positive")
        let plan = Qwen3NextLayerPlan(configuration)
        self.layerPlan = plan

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        _layers.wrappedValue = (0 ..< plan.count).map { layerIndex in
            Qwen3NextDecoderLayer(
                configuration,
                layerIndex: layerIndex,
                layerPlan: plan
            )
        }
        _norm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        super.init()
    }

    func hiddenStates(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs)
        let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let index = layerPlan.firstFullAttentionIndex {
            attentionMask = createAttentionMask(h: hidden, cache: cacheValue(in: cacheArray, at: index))
        } else {
            attentionMask = .none
        }

        let ssmMask: MLXArray?
        if let index = layerPlan.firstLinearIndex {
            ssmMask = createSSMMask(h: hidden, cache: cacheValue(in: cacheArray, at: index) as? MambaCache)
        } else {
            ssmMask = nil
        }

        for (layerIndex, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                attentionMask: layer.isLinear ? .none : attentionMask,
                ssmMask: layer.isLinear ? ssmMask : nil,
                cache: cacheValue(in: cacheArray, at: layerIndex)
            )
        }

        return hidden
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        norm(hiddenStates(inputs, cache: cache))
    }

    private func cacheValue(in cache: [KVCache?], at index: Int) -> KVCache? {
        cache.indices.contains(index) ? cache[index] : nil
    }
}

internal final class Qwen3NextModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: Qwen3NextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: Qwen3NextConfiguration

    internal init(_ configuration: Qwen3NextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0 ..< configuration.hiddenLayers).map { _ in configuration.kvHeads }
        _model.wrappedValue = Qwen3NextModelInner(configuration)

        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.hiddenStates(inputs, cache: cache)
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
        model.layerPlan.kinds.map { kind in
            switch kind {
            case .linearAttention:
                MambaCache()
            case .fullAttention:
                KVCacheSimple()
            }
        }
    }

    internal func makeCache() -> [KVCache] {
        newCache(parameters: nil)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Qwen3NextSanitizerPlan(configuration).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}

// MARK: - LoRA

extension Qwen3NextModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.compactMap { layer in
            guard let attention = layer.selfAttn else {
                return nil
            }
            return (attention, ["q_proj", "v_proj"])
        }
    }
}
