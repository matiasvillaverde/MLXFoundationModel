//
//  Gemma4Text.swift
//  MLXSession
//
//  Text-only Gemma 4 implementation based on mlx-vlm's Gemma 4 language model.
//

import Foundation
import MLX
import MLXFast
import MLXNN

internal struct Gemma4TextConfiguration: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let headDim: Int
    let globalHeadDim: Int?
    let rmsNormEps: Float
    let vocabSize: Int
    let vocabSizePerLayerInput: Int
    let numKeyValueHeads: Int
    let numGlobalKeyValueHeads: Int?
    let numKvSharedLayers: Int
    let hiddenSizePerLayerInput: Int
    let slidingWindow: Int
    let maxPositionEmbeddings: Int
    let finalLogitSoftcapping: Float?
    let layerTypes: [String]
    let ropeParameters: [String: Gemma4RopeParameters]
    let ropeTraditional: Bool
    let attentionKEqualsV: Bool
    let useDoubleWideMLP: Bool
    let enableMoeBlock: Bool
    let useSecondMLPBlock: Bool
    let numExperts: Int?
    let topKExperts: Int?
    let moeIntermediateSize: Int?
    let slidingWindowPattern: Int
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case ropeTraditional = "rope_traditional"
        case attentionKEqualsV = "attention_k_eq_v"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoeBlock = "enable_moe_block"
        case useSecondMLPBlock = "use_second_mlp_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case slidingWindowPattern = "sliding_window_pattern"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? vocabSize
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        numGlobalKeyValueHeads = try container.decodeIfPresent(
            Int.self, forKey: .numGlobalKeyValueHeads)
        numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        finalLogitSoftcapping = try container.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
        ropeParameters =
            try container.decodeIfPresent(
                [String: Gemma4RopeParameters].self, forKey: .ropeParameters)
            ?? Gemma4RopeParameters.defaults
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        attentionKEqualsV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKEqualsV) ?? false
        useDoubleWideMLP =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMLP) ?? true
        enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        useSecondMLPBlock =
            try container.decodeIfPresent(Bool.self, forKey: .useSecondMLPBlock) ?? false
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        if let decodedLayerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        {
            layerTypes = Self.expandLayerTypes(decodedLayerTypes, count: numHiddenLayers)
        } else {
            let slidingCount = max(0, slidingWindowPattern - 1)
            let pattern = Array(repeating: "sliding_attention", count: slidingCount) +
                ["full_attention"]
            layerTypes = Self.expandLayerTypes(pattern, count: numHiddenLayers)
        }

        if enableMoeBlock {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [CodingKeys.enableMoeBlock],
                    debugDescription:
                        "Gemma 4 MoE checkpoints are not supported by MLXSession yet. " +
                        "Use dense E2B/E4B Gemma 4 models."
                )
            )
        }

        if useSecondMLPBlock || numExperts != nil || topKExperts != nil ||
            moeIntermediateSize != nil {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [CodingKeys.numExperts],
                    debugDescription:
                        "Gemma 4 expert layers are not supported by MLXSession yet. " +
                        "Use dense E2B/E4B Gemma 4 models."
                )
            )
        }

        if !tieWordEmbeddings {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [CodingKeys.tieWordEmbeddings],
                    debugDescription:
                        "Gemma 4 untied output embeddings are not supported by MLXSession yet."
                )
            )
        }

        if numKvSharedLayers > numHiddenLayers {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [CodingKeys.numKvSharedLayers],
                    debugDescription:
                        "Gemma 4 num_kv_shared_layers cannot exceed num_hidden_layers."
                )
            )
        }
    }

    private static func expandLayerTypes(_ layerTypes: [String], count: Int) -> [String] {
        guard !layerTypes.isEmpty else {
            return Array(repeating: "full_attention", count: count)
        }
        return (0 ..< count).map { layerTypes[$0 % layerTypes.count] }
    }
}

internal struct Gemma4RopeParameters: Codable, Sendable {
    let ropeTheta: Float
    let ropeType: String
    let partialRotaryFactor: Float
    let factor: Float

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case partialRotaryFactor = "partial_rotary_factor"
        case factor
    }

    init(
        ropeTheta: Float,
        ropeType: String = "default",
        partialRotaryFactor: Float = 1.0,
        factor: Float = 1.0
    ) {
        self.ropeTheta = ropeTheta
        self.ropeType = ropeType
        self.partialRotaryFactor = partialRotaryFactor
        self.factor = factor
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
        partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        factor = try container.decodeIfPresent(Float.self, forKey: .factor) ?? 1.0
    }

    static let defaults: [String: Gemma4RopeParameters] = [
        "full_attention": Gemma4RopeParameters(
            ropeTheta: 1_000_000,
            ropeType: "proportional",
            partialRotaryFactor: 0.25
        ),
        "sliding_attention": Gemma4RopeParameters(
            ropeTheta: 10_000,
            ropeType: "default"
        )
    ]
}

private final class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(input, weight: MLXArray.mlxNone, eps: eps)
    }
}

private final class Gemma4ProportionalRoPE: Module {
    let dimensions: Int
    let traditional: Bool
    let rotatedDimensions: Int
    private let frequencies: MLXArray?

    init(dimensions: Int, traditional: Bool, base: Float, parameters: Gemma4RopeParameters) {
        self.dimensions = dimensions
        self.traditional = traditional

        let ropeAngles = Int(parameters.partialRotaryFactor * Float(dimensions) / 2)
        self.rotatedDimensions = 2 * ropeAngles

        if rotatedDimensions > 0 {
            let exponent =
                MLXArray(stride(from: 0, to: rotatedDimensions, by: 2)).asType(.float32)
                / Float(dimensions)
            self.frequencies = parameters.factor * (base ** exponent)
        } else {
            self.frequencies = nil
        }

        super.init()
    }

    func callAsFunction(_ input: MLXArray, offset: Int = 0) -> MLXArray {
        guard rotatedDimensions > 0, let frequencies else {
            return input
        }

        let lastDimension = input.shape.last ?? 0
        let head = input[.ellipsis, ..<dimensions]
        let tail: MLXArray? = lastDimension > dimensions ? input[.ellipsis, dimensions...] : nil
        let half = dimensions / 2
        let rotatedHalf = rotatedDimensions / 2

        let left = head[.ellipsis, ..<half]
        let right = head[.ellipsis, half...]
        let rotated = concatenated(
            [
                left[.ellipsis, ..<rotatedHalf],
                right[.ellipsis, ..<rotatedHalf]
            ],
            axis: -1
        )
        let encoded = MLXFast.RoPE(
            rotated,
            dimensions: rotatedDimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: frequencies
        )

        let encodedLeft = encoded[.ellipsis, ..<rotatedHalf]
        let encodedRight = encoded[.ellipsis, rotatedHalf...]
        let finalLeft = concatenated(
            [encodedLeft, left[.ellipsis, rotatedHalf...]],
            axis: -1
        )
        let finalRight = concatenated(
            [encodedRight, right[.ellipsis, rotatedHalf...]],
            axis: -1
        )
        let encodedHead = concatenated([finalLeft, finalRight], axis: -1)

        if let tail {
            return concatenated([encodedHead, tail], axis: -1)
        }
        return encodedHead
    }
}

private enum Gemma4RoPE {
    case standard(RoPE)
    case proportional(Gemma4ProportionalRoPE)

    init(dimensions: Int, traditional: Bool, parameters: Gemma4RopeParameters) {
        if parameters.ropeType == "proportional" {
            self = .proportional(
                Gemma4ProportionalRoPE(
                    dimensions: dimensions,
                    traditional: traditional,
                    base: parameters.ropeTheta,
                    parameters: parameters
                )
            )
        } else {
            self = .standard(
                RoPE(
                    dimensions: dimensions,
                    traditional: traditional,
                    base: parameters.ropeTheta
                )
            )
        }
    }

    func callAsFunction(_ input: MLXArray, offset: Int) -> MLXArray {
        switch self {
        case .standard(let rope):
            return rope(input, offset: offset)
        case .proportional(let rope):
            return rope(input, offset: offset)
        }
    }
}

private final class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIndex: Int) {
        let firstKVSharedLayer = config.numHiddenLayers - config.numKvSharedLayers
        let isKVSharedLayer = layerIndex >= firstKVSharedLayer && firstKVSharedLayer > 0
        let usesDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let intermediateSize = config.intermediateSize * (usesDoubleWide ? 2 : 1)

        _gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(input)) * upProj(input))
    }
}

private final class Gemma4Attention: Module {
    let layerType: String
    let isSliding: Bool
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let useKEqualsV: Bool
    let scale: Float = 1.0
    let rope: Gemma4RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale

    init(_ config: Gemma4TextConfiguration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self.isSliding = layerType == "sliding_attention"
        self.headDim =
            if !isSliding, let globalHeadDim = config.globalHeadDim {
                globalHeadDim
            } else {
                config.headDim
            }
        self.numHeads = config.numAttentionHeads
        self.useKEqualsV = config.attentionKEqualsV && !isSliding
        if useKEqualsV, let globalKVHeads = config.numGlobalKeyValueHeads {
            self.numKVHeads = globalKVHeads
        } else {
            self.numKVHeads = config.numKeyValueHeads
        }

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        if !useKEqualsV {
            _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        }
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        let ropeParameters =
            config.ropeParameters[isSliding ? "sliding_attention" : "full_attention"]
            ?? Gemma4RopeParameters.defaults[isSliding ? "sliding_attention" : "full_attention"]!
        self.rope = Gemma4RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            parameters: ropeParameters
        )

        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil,
        sharedOffset: Int? = nil
    ) -> (output: MLXArray, kv: (keys: MLXArray, values: MLXArray), offset: Int) {
        let (batchSize, sequenceLength, _) = (input.dim(0), input.dim(1), input.dim(2))
        let offset = sharedOffset ?? cache?.offset ?? 0

        var queries = qProj(input).reshaped(batchSize, sequenceLength, numHeads, headDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        let keys: MLXArray
        let values: MLXArray
        let attentionOutput: MLXArray
        if let sharedKV {
            keys = sharedKV.keys
            values = sharedKV.values
            let adjustedMask = adjustedMask(mask, keySequenceLength: keys.shape[keys.shape.count - 2])
            attentionOutput = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: adjustedMask
            )
        } else {
            var projectedKeys = kProj(input).reshaped(
                batchSize,
                sequenceLength,
                numKVHeads,
                headDim
            )
            var projectedValues =
                if useKEqualsV {
                    projectedKeys
                } else {
                    vProj!(input).reshaped(batchSize, sequenceLength, numKVHeads, headDim)
                }

            projectedKeys = kNorm(projectedKeys)
                .transposed(0, 2, 1, 3)
            projectedKeys = rope(projectedKeys, offset: offset)

            projectedValues = vNorm(projectedValues)
                .transposed(0, 2, 1, 3)

            let result = attentionWithCacheUpdateReturningKV(
                queries: queries,
                keys: projectedKeys,
                values: projectedValues,
                cache: cache,
                scale: scale,
                mask: mask,
                maskForKeySequenceLength: { keySequenceLength in
                    self.adjustedMask(mask, keySequenceLength: keySequenceLength)
                }
            )
            keys = result.keys
            values = result.values
            attentionOutput = result.output
        }

        let output = attentionOutput
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return (oProj(output), (keys, values), offset)
    }

    private func adjustedMask(
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
        keySequenceLength: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard case .array(let maskArray) = mask,
            maskArray.shape.last != keySequenceLength else {
            return mask
        }
        return .array(maskArray[.ellipsis, (-keySequenceLength)...])
    }
}

private final class Gemma4DecoderLayer: Module {
    let layerType: String
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        _selfAttention.wrappedValue = Gemma4Attention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = Gemma4MLP(config, layerIndex: layerIndex)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )

        if config.hiddenSizePerLayerInput > 0 {
            _perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSizePerLayerInput,
                bias: false
            )
            _perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput,
                config.hiddenSize,
                bias: false
            )
            _postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
        }
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        perLayerInput: MLXArray?,
        sharedKV: (keys: MLXArray, values: MLXArray)?,
        sharedOffset: Int?
    ) -> (output: MLXArray, kv: (keys: MLXArray, values: MLXArray), offset: Int) {
        var hidden = input

        let attentionResult = selfAttention(
            inputLayerNorm(hidden),
            mask: mask,
            cache: cache,
            sharedKV: sharedKV,
            sharedOffset: sharedOffset
        )
        hidden = hidden + postAttentionLayerNorm(attentionResult.output)

        let residual = hidden
        hidden = preFeedforwardLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = postFeedforwardLayerNorm(hidden)
        hidden = residual + hidden

        if let perLayerInputGate,
            let perLayerProjection,
            let postPerLayerInputNorm,
            let perLayerInput {
            let residual = hidden
            var gated = perLayerInputGate(hidden)
            gated = geluApproximate(gated) * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            hidden = residual + gated
        }

        return (hidden * layerScalar, attentionResult.kv, attentionResult.offset)
    }
}

private final class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let firstKVSharedLayerIndex: Int
    let previousKVs: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    private let embedScale: Float
    private let embedTokensPerLayerScale: Float
    private let perLayerProjectionScale: Float
    private let perLayerInputScale: Float = pow(2.0, -0.5)

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIndex = config.numHiddenLayers - config.numKvSharedLayers
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(config.hiddenSizePerLayerInput), 0.5)
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)

        var previousKVs = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            var kvsByType: [String: Int] = [:]
            for index in 0 ..< firstKVSharedLayerIndex {
                kvsByType[config.layerTypes[index]] = index
            }
            for index in firstKVSharedLayerIndex ..< config.numHiddenLayers {
                previousKVs[index] = kvsByType[config.layerTypes[index]] ?? index
            }
        }
        self.previousKVs = previousKVs

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hiddenSizePerLayerInput > 0 {
            _embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput
            )
            _perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.numHiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            _perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput,
                eps: config.rmsNormEps
            )
        }

        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        config.layerTypes[..<firstKVSharedLayerIndex].map { layerType in
            if layerType == "full_attention" {
                KVCacheSimple()
            } else {
                RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        hidden = (hidden * MLXArray(embedScale, dtype: .float32)).asType(hidden.dtype)

        let perLayerInputs = makePerLayerInputs(inputIDs: inputs, hiddenStates: hidden)
        let cacheArray = paddedCache(cache)
        let masks = makeMasks(hiddenStates: hidden, cache: cacheArray)

        var intermediates = Array<(keys: MLXArray, values: MLXArray, offset: Int)?>(repeating: nil, count: layers.count)
        for (index, layer) in layers.enumerated() {
            let previousIndex = previousKVs[index]
            let shared = intermediates[previousIndex]
            let perLayerInput = perLayerInputs?[0..., 0..., index, 0...]
            let result = layer(
                hidden,
                mask: masks[index],
                cache: cacheArray[index],
                perLayerInput: perLayerInput,
                sharedKV: shared.map { ($0.keys, $0.values) },
                sharedOffset: shared?.offset
            )
            hidden = result.output
            intermediates[index] = (result.kv.keys, result.kv.values, result.offset)
        }

        return norm(hidden)
    }

    private func makePerLayerInputs(inputIDs: MLXArray, hiddenStates: MLXArray) -> MLXArray? {
        guard config.hiddenSizePerLayerInput > 0,
            let embedTokensPerLayer,
            let perLayerModelProjection,
            let perLayerProjectionNorm else {
            return nil
        }

        let perLayerInputsMask = logicalAnd(
            inputIDs .>= 0,
            inputIDs .< config.vocabSizePerLayerInput
        )
        let tokenIDs = MLX.where(perLayerInputsMask, inputIDs, MLXArray.zeros(like: inputIDs))
        var tokenInputs = embedTokensPerLayer(tokenIDs)
        tokenInputs = (tokenInputs * MLXArray(embedTokensPerLayerScale, dtype: .float32))
            .asType(tokenInputs.dtype)
        tokenInputs = tokenInputs.reshaped(
            Array(inputIDs.shape) + [
                config.numHiddenLayers,
                config.hiddenSizePerLayerInput
            ]
        )

        var projected = perLayerModelProjection(hiddenStates)
        projected = projected * MLXArray(perLayerProjectionScale, dtype: hiddenStates.dtype)
        projected = projected.reshaped(
            Array(hiddenStates.shape.dropLast()) + [
                config.numHiddenLayers,
                config.hiddenSizePerLayerInput
            ]
        )
        projected = perLayerProjectionNorm(projected)

        return (projected + tokenInputs) * MLXArray(perLayerInputScale, dtype: hiddenStates.dtype)
    }

    private func paddedCache(_ cache: [KVCache]?) -> [KVCache?] {
        var cacheArray = cache?.map { Optional($0) } ?? []
        if cacheArray.count < layers.count {
            cacheArray.append(contentsOf: Array(repeating: nil, count: layers.count - cacheArray.count))
        }
        return cacheArray
    }

    private func makeMasks(
        hiddenStates: MLXArray,
        cache: [KVCache?]
    ) -> [MLXFast.ScaledDotProductAttentionMaskMode] {
        var masksByType: [String: MLXFast.ScaledDotProductAttentionMaskMode] = [:]
        var masks: [MLXFast.ScaledDotProductAttentionMaskMode] = []

        for (layer, layerCache) in zip(layers, cache) {
            if let mask = masksByType[layer.layerType] {
                masks.append(mask)
                continue
            }

            let windowSize = layer.layerType == "sliding_attention" ? config.slidingWindow : nil
            let mask = makeMask(hiddenStates: hiddenStates, cache: layerCache, windowSize: windowSize)
            masksByType[layer.layerType] = mask
            masks.append(mask)
        }

        return masks
    }

    private func makeMask(
        hiddenStates: MLXArray,
        cache: KVCache?,
        windowSize: Int?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let tokenCount = hiddenStates.dim(1)
        guard tokenCount > 1 else {
            return .none
        }

        let rawOffset = cache?.offset ?? 0
        let offset =
            if let windowSize {
                min(windowSize, rawOffset)
            } else {
                rawOffset
            }
        return .array(createCausalMask(n: tokenCount, offset: offset, windowSize: windowSize))
    }
}

private final class Gemma4LanguageModel: Module {
    let config: Gemma4TextConfiguration

    @ModuleInfo(key: "model") var model: Gemma4TextModelInner

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        _model.wrappedValue = Gemma4TextModelInner(config)
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.newCache(parameters: parameters)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var output = model(inputs, cache: cache)
        output = model.embedTokens.asLinear(output)
        if let softcap = config.finalLogitSoftcapping {
            output = tanh(output / softcap) * softcap
        }
        return output
    }
}

internal final class Gemma4Model: Module, LLMModel {
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4LanguageModel

    let config: Gemma4TextConfiguration
    var vocabularySize: Int { config.vocabSize }

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        _languageModel.wrappedValue = Gemma4LanguageModel(config)
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, pair in
            guard let key = sanitizeWeightKey(pair.key) else {
                return
            }
            result[key] = pair.value
        }
    }

    private func sanitizeWeightKey(_ originalKey: String) -> String? {
        var key = originalKey
        if key.hasPrefix("model.") {
            key.removeFirst("model.".count)
        }

        guard !shouldDropWeight(key) else {
            return nil
        }

        if key.hasPrefix("language_model.model.") {
            return key
        }

        if key.hasPrefix("language_model.") {
            let rest = String(key.dropFirst("language_model.".count))
            if rest.hasPrefix("lm_head.") || rest.hasPrefix("model.lm_head.") {
                return nil
            }
            if rest.hasPrefix("model.") {
                return "language_model.\(rest)"
            }
            return "language_model.model.\(rest)"
        }

        if key.hasPrefix("text_model.") {
            return "language_model.model.\(key.dropFirst("text_model.".count))"
        }

        return nil
    }

    private func shouldDropWeight(_ key: String) -> Bool {
        let droppedPrefixes = [
            "vision_tower.",
            "audio_tower.",
            "embed_vision.",
            "embed_audio.",
            "multi_modal_projector.",
            "language_model.lm_head.",
            "language_model.model.lm_head."
        ]
        if droppedPrefixes.contains(where: key.hasPrefix) {
            return true
        }

        let droppedFragments = [
            "input_max",
            "input_min",
            "output_max",
            "output_min",
            "rotary_emb.inv_freq"
        ]
        return droppedFragments.contains(where: key.contains)
    }
}

extension Gemma4Model: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        languageModel.model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
