import Foundation
import MLX
import MLXFast
import MLXNN

internal enum Plamo2LayerKind: String, Sendable, Equatable {
    case mamba
    case attention
}

internal struct Plamo2Configuration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var tieWordEmbeddings: Bool
    var attentionHeads: Int
    var keyValueHeads: Int
    var headSize: Int
    var maxPositionEmbeddings: Int
    var attentionWindowSize: Int
    var mambaStateSize: Int
    var mambaConvKernel: Int
    var mambaHeads: Int
    var mambaStep: Int
    var mambaChunkSize: Int
    var mambaEnabled: Bool
    var intermediateSize: Int
    var vocabularySize: Int

    internal init(
        modelType: String = "plamo2",
        hiddenSize: Int = 4_096,
        hiddenLayers: Int = 32,
        rmsNormEps: Float = 1e-6,
        tieWordEmbeddings: Bool = true,
        attentionHeads: Int = 32,
        keyValueHeads: Int = 4,
        headSize: Int = 128,
        maxPositionEmbeddings: Int = 2_048,
        attentionWindowSize: Int = 2_048,
        mambaStateSize: Int = 64,
        mambaConvKernel: Int = 4,
        mambaHeads: Int = 64,
        mambaStep: Int = 2,
        mambaChunkSize: Int = 256,
        mambaEnabled: Bool = true,
        intermediateSize: Int = 13_312,
        vocabularySize: Int = 32_000
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.rmsNormEps = rmsNormEps
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads
        self.headSize = headSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionWindowSize = attentionWindowSize
        self.mambaStateSize = mambaStateSize
        self.mambaConvKernel = mambaConvKernel
        self.mambaHeads = mambaHeads
        self.mambaStep = mambaStep
        self.mambaChunkSize = mambaChunkSize
        self.mambaEnabled = mambaEnabled
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headSize = "hidden_size_per_head"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionWindowSize = "attention_window_size"
        case slidingWindow = "sliding_window"
        case mambaStateSize = "mamba_d_state"
        case mambaConvKernel = "mamba_d_conv"
        case mambaHeads = "mamba_num_heads"
        case mambaStep = "mamba_step"
        case mambaChunkSize = "mamba_chunk_size"
        case mambaEnabled = "mamba_enabled"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "plamo2",
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? 4_096,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? 32,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-6,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? 32,
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads)
                ?? 4,
            headSize: try container.decodeIfPresent(Int.self, forKey: .headSize) ?? 128,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 2_048,
            attentionWindowSize: try container.decodeIfPresent(
                Int.self,
                forKey: .attentionWindowSize
            ) ?? container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2_048,
            mambaStateSize: try container.decodeIfPresent(Int.self, forKey: .mambaStateSize)
                ?? 64,
            mambaConvKernel: try container.decodeIfPresent(Int.self, forKey: .mambaConvKernel)
                ?? 4,
            mambaHeads: try container.decodeIfPresent(Int.self, forKey: .mambaHeads)
                ?? 64,
            mambaStep: try container.decodeIfPresent(Int.self, forKey: .mambaStep) ?? 2,
            mambaChunkSize: try container.decodeIfPresent(Int.self, forKey: .mambaChunkSize)
                ?? 256,
            mambaEnabled: try container.decodeIfPresent(Bool.self, forKey: .mambaEnabled)
                ?? true,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? 13_312,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 32_000
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(keyValueHeads, forKey: .keyValueHeads)
        try container.encode(headSize, forKey: .headSize)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(attentionWindowSize, forKey: .attentionWindowSize)
        try container.encode(mambaStateSize, forKey: .mambaStateSize)
        try container.encode(mambaConvKernel, forKey: .mambaConvKernel)
        try container.encode(mambaHeads, forKey: .mambaHeads)
        try container.encode(mambaStep, forKey: .mambaStep)
        try container.encode(mambaChunkSize, forKey: .mambaChunkSize)
        try container.encode(mambaEnabled, forKey: .mambaEnabled)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(vocabularySize, forKey: .vocabularySize)
    }
}

internal struct Plamo2LayerPlan: Sendable, Equatable {
    internal let kinds: [Plamo2LayerKind]
    internal let firstAttentionIndex: Int?
    internal let firstMambaIndex: Int?

    internal init(_ config: Plamo2Configuration) {
        precondition(config.hiddenLayers > 0, "Plamo2 num_hidden_layers must be positive")
        precondition(config.mambaStep > 1, "Plamo2 mamba_step must be greater than one")

        self.kinds = (0 ..< config.hiddenLayers).map { layerIndex in
            Self.usesMamba(config, layerIndex: layerIndex) ? .mamba : .attention
        }
        self.firstAttentionIndex = kinds.firstIndex(of: .attention)
        self.firstMambaIndex = kinds.firstIndex(of: .mamba)
    }

    internal var kvHeads: [Int] {
        kinds.map { $0 == .attention ? 1 : 0 }
    }

    internal func kind(at index: Int) -> Plamo2LayerKind {
        kinds[index]
    }

    private static func usesMamba(_ config: Plamo2Configuration, layerIndex: Int) -> Bool {
        guard config.mambaEnabled else { return false }
        if config.hiddenLayers <= config.mambaStep / 2 {
            return layerIndex != config.hiddenLayers - 1
        }
        return layerIndex % config.mambaStep != config.mambaStep / 2
    }
}

internal struct Plamo2AttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryDimensions: Int
    internal let keyDimensions: Int
    internal let valueDimensions: Int
    internal let projectionDimensions: Int
    internal let scale: Float

    internal init(_ config: Plamo2Configuration) {
        precondition(config.hiddenSize > 0, "Plamo2 hidden_size must be positive")
        precondition(config.attentionHeads > 0, "Plamo2 num_attention_heads must be positive")
        precondition(config.keyValueHeads > 0, "Plamo2 num_key_value_heads must be positive")
        precondition(config.headSize > 0, "Plamo2 hidden_size_per_head must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.keyValueHeads),
            "Plamo2 attention heads must group key/value heads"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.headSize = config.headSize
        self.queryDimensions = queryHeads * headSize
        self.keyDimensions = keyValueHeads * headSize
        self.valueDimensions = keyValueHeads * headSize
        self.projectionDimensions = queryDimensions + keyDimensions + valueDimensions
        self.scale = pow(Float(headSize), -0.5)
    }
}

internal struct Plamo2MambaLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let heads: Int
    internal let headSize: Int
    internal let stateSize: Int
    internal let convolutionKernel: Int
    internal let intermediateSize: Int
    internal let timeStepDimensions: Int
    internal let inputProjectionDimensions: Int
    internal let stateProjectionDimensions: Int

    internal init(_ config: Plamo2Configuration) {
        precondition(config.hiddenSize > 0, "Plamo2 hidden_size must be positive")
        precondition(config.mambaHeads > 0, "Plamo2 mamba_num_heads must be positive")
        precondition(config.headSize > 0, "Plamo2 hidden_size_per_head must be positive")
        precondition(config.mambaStateSize > 0, "Plamo2 mamba_d_state must be positive")
        precondition(config.mambaConvKernel > 1, "Plamo2 mamba_d_conv must be greater than one")

        self.hiddenSize = config.hiddenSize
        self.heads = config.mambaHeads
        self.headSize = config.headSize
        self.stateSize = config.mambaStateSize
        self.convolutionKernel = config.mambaConvKernel
        self.intermediateSize = heads * headSize
        self.timeStepDimensions = max(64, hiddenSize / 16)
        self.inputProjectionDimensions = intermediateSize * 2
        self.stateProjectionDimensions = timeStepDimensions + 2 * stateSize
    }
}

private final class Plamo2RMSNorm: Module {
    @ParameterInfo(key: "weight") private var weight: MLXArray

    private let eps: Float
    private let offset: Float

    init(hiddenSize: Int, eps: Float, offset: Float = 1) {
        self.eps = eps
        self.offset = offset
        _weight.wrappedValue = MLXArray.zeros([hiddenSize])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(
            hiddenStates,
            weight: weight + MLXArray(offset).asType(weight.dtype),
            eps: eps
        )
    }
}

private protocol Plamo2MixerLayer {
    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

private final class Plamo2Attention: Module, Plamo2MixerLayer {
    private let layout: Plamo2AttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "qkv_proj") private var qkvProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ParameterInfo(key: "q_weight") private var queryNormWeight: MLXArray
    @ParameterInfo(key: "k_weight") private var keyNormWeight: MLXArray

    init(_ config: Plamo2Configuration) {
        let layout = Plamo2AttentionLayout(config)
        self.layout = layout
        self.rope = initializeRope(
            dims: layout.headSize,
            base: 10_000,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        _qkvProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.projectionDimensions,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryDimensions,
            layout.hiddenSize,
            bias: false
        )
        _queryNormWeight.wrappedValue = MLXArray.ones([layout.queryHeads, layout.headSize])
        _keyNormWeight.wrappedValue = MLXArray.ones([layout.keyValueHeads, layout.headSize])
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)
        let qkv = qkvProjection(hiddenStates)
        let parts = split(
            qkv,
            indices: [layout.queryDimensions, layout.queryDimensions + layout.keyDimensions],
            axis: -1
        )

        var queries = parts[0]
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = parts[1]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = parts[2]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        queries = queryNormWeight[0..., .newAxis, 0...].asType(queries.dtype)
            * MLXFast.rmsNorm(queries, weight: MLXArray.mlxNone, eps: 1e-6)
        keys = keyNormWeight[0..., .newAxis, 0...].asType(keys.dtype)
            * MLXFast.rmsNorm(keys, weight: MLXArray.mlxNone, eps: 1e-6)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        return outputProjection(
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, layout.queryDimensions)
        )
    }
}

private final class Plamo2Mamba: Module, Plamo2MixerLayer {
    private let layout: Plamo2MambaLayout
    private let eps: Float

    @ModuleInfo(key: "in_proj") private var inputProjection: Linear
    @ModuleInfo(key: "conv1d") private var convolution: Conv1d
    @ModuleInfo(key: "bcdt_proj") private var stateProjection: Linear
    @ModuleInfo(key: "dt_proj") private var timeStepProjection: Linear
    @ModuleInfo(key: "out_proj") private var outputProjection: Linear

    @ParameterInfo(key: "dt_bias") private var timeStepBias: MLXArray
    @ParameterInfo(key: "A_log") private var stateLog: MLXArray
    @ParameterInfo(key: "D") private var residualScale: MLXArray
    @ParameterInfo(key: "dt_norm_weight") private var timeStepNormWeight: MLXArray
    @ParameterInfo(key: "B_norm_weight") private var inputStateNormWeight: MLXArray
    @ParameterInfo(key: "C_norm_weight") private var outputStateNormWeight: MLXArray

    init(_ config: Plamo2Configuration) {
        let layout = Plamo2MambaLayout(config)
        self.layout = layout
        self.eps = config.rmsNormEps

        _inputProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.inputProjectionDimensions,
            bias: false
        )
        _convolution.wrappedValue = Conv1d(
            inputChannels: layout.intermediateSize,
            outputChannels: layout.intermediateSize,
            kernelSize: layout.convolutionKernel,
            padding: 0,
            groups: layout.intermediateSize,
            bias: false
        )
        _stateProjection.wrappedValue = Linear(
            layout.intermediateSize,
            layout.stateProjectionDimensions,
            bias: false
        )
        _timeStepProjection.wrappedValue = Linear(
            layout.timeStepDimensions,
            layout.heads,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.intermediateSize,
            layout.hiddenSize,
            bias: false
        )

        _timeStepBias.wrappedValue = MLXArray.zeros([layout.heads])
        _stateLog.wrappedValue = log((MLXArray(0 ..< layout.heads).asType(.float32) + 1))
        _residualScale.wrappedValue = MLXArray.ones([layout.heads])
        _timeStepNormWeight.wrappedValue = MLXArray.ones([layout.timeStepDimensions])
        _inputStateNormWeight.wrappedValue = MLXArray.ones([layout.stateSize])
        _outputStateNormWeight.wrappedValue = MLXArray.ones([layout.stateSize])
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)
        let mambaCache = cache as? MambaCache

        let projected = inputProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.heads, layout.headSize * 2)
        let projectedParts = split(projected, indices: [layout.headSize], axis: -1)
        let gate = projectedParts[0].flattened(start: 2)
        var convInput = projectedParts[1].flattened(start: 2)

        if let ssmMask {
            convInput = MLX.where(
                expandedDimensions(ssmMask, axis: -1),
                convInput,
                MLXArray.zeros(like: convInput)
            )
        }

        let convOutput = applyConvolution(convInput, cache: mambaCache)
        let stateProjection = stateProjection(convOutput)
        let stateParts = split(
            stateProjection,
            indices: [layout.stateSize, layout.stateSize * 2],
            axis: -1
        )

        let inputState = MLXFast.rmsNorm(
            stateParts[0],
            weight: inputStateNormWeight,
            eps: eps
        ).reshaped(batchSize, tokenCount, 1, layout.stateSize)
        let outputState = MLXFast.rmsNorm(
            stateParts[1],
            weight: outputStateNormWeight,
            eps: eps
        ).reshaped(batchSize, tokenCount, 1, layout.stateSize)
        let timeSteps = timeStepProjection(
            MLXFast.rmsNorm(stateParts[2], weight: timeStepNormWeight, eps: eps)
        ).reshaped(batchSize, tokenCount, layout.heads)

        let ssmInput = convOutput.reshaped(
            batchSize,
            tokenCount,
            layout.heads,
            layout.headSize
        )
        let (output, nextState) = ssmUpdate(
            hiddenStates: ssmInput,
            ALog: stateLog,
            B: inputState,
            C: outputState,
            D: residualScale,
            dt: timeSteps,
            dtBias: timeStepBias,
            state: mambaCache?[1],
            mask: ssmMask
        )

        if let mambaCache {
            mambaCache[1] = nextState
            mambaCache.offset += tokenCount
        }

        return outputProjection(silu(gate) * output.flattened(start: 2))
    }

    private func applyConvolution(_ input: MLXArray, cache: MambaCache?) -> MLXArray {
        let batchSize = input.dim(0)
        let stateLength = layout.convolutionKernel - 1
        let state = cache?[0] ?? MLXArray.zeros(
            [batchSize, stateLength, layout.intermediateSize],
            dtype: input.dtype
        )
        let paddedInput = concatenated([state, input], axis: 1)

        if let cache {
            let end = paddedInput.dim(1)
            cache[0] = paddedInput[0..., (end - stateLength) ..< end, 0...]
        }

        return silu(convolution(paddedInput))
    }
}

private final class Plamo2MLP: Module {
    @ModuleInfo(key: "gate_up_proj") private var gateUpProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: Plamo2Configuration) {
        _gateUpProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize * 2,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let parts = split(gateUpProjection(input), parts: 2, axis: -1)
        return downProjection(silu(parts[0]) * parts[1])
    }
}

private final class Plamo2Block: Module {
    internal let layerKind: Plamo2LayerKind

    @ModuleInfo(key: "mixer") fileprivate var mixer: Module & Plamo2MixerLayer
    @ModuleInfo(key: "mlp") private var mlp: Plamo2MLP
    @ModuleInfo(key: "pre_mixer_norm") private var preMixerNorm: Plamo2RMSNorm
    @ModuleInfo(key: "post_mixer_norm") private var postMixerNorm: Plamo2RMSNorm
    @ModuleInfo(key: "pre_mlp_norm") private var preMLPNorm: Plamo2RMSNorm
    @ModuleInfo(key: "post_mlp_norm") private var postMLPNorm: Plamo2RMSNorm

    init(_ config: Plamo2Configuration, layerIndex: Int, layerPlan: Plamo2LayerPlan) {
        self.layerKind = layerPlan.kind(at: layerIndex)
        switch layerKind {
        case .mamba:
            _mixer.wrappedValue = Plamo2Mamba(config)
        case .attention:
            _mixer.wrappedValue = Plamo2Attention(config)
        }
        _mlp.wrappedValue = Plamo2MLP(config)
        _preMixerNorm.wrappedValue = Plamo2RMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps,
            offset: 1
        )
        _postMixerNorm.wrappedValue = Plamo2RMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps,
            offset: 0.2
        )
        _preMLPNorm.wrappedValue = Plamo2RMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps,
            offset: 1
        )
        _postMLPNorm.wrappedValue = Plamo2RMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps,
            offset: Float(1 / pow(5.0, 1.5))
        )
        super.init()
    }

    internal var isMamba: Bool {
        layerKind == .mamba
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let mixerOutput = mixer(
            preMixerNorm(hiddenStates),
            attentionMask: attentionMask,
            ssmMask: ssmMask,
            cache: cache
        )
        let afterMixer = hiddenStates + postMixerNorm(mixerOutput)
        return afterMixer + postMLPNorm(mlp(preMLPNorm(afterMixer)))
    }
}

private final class Plamo2Decoder: Module {
    @ModuleInfo fileprivate var layers: [Plamo2Block]
    private let layerPlan: Plamo2LayerPlan

    init(_ config: Plamo2Configuration) {
        let layerPlan = Plamo2LayerPlan(config)
        self.layerPlan = layerPlan
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIndex in
            Plamo2Block(config, layerIndex: layerIndex, layerPlan: layerPlan)
        }
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hiddenStates = hiddenStates
        let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let index = layerPlan.firstAttentionIndex {
            attentionMask = createAttentionMask(h: hiddenStates, cache: cacheValue(in: cacheArray, at: index))
        } else {
            attentionMask = .none
        }

        let ssmMask: MLXArray?
        if let index = layerPlan.firstMambaIndex {
            ssmMask = createSSMMask(
                h: hiddenStates,
                cache: cacheValue(in: cacheArray, at: index) as? MambaCache
            )
        } else {
            ssmMask = nil
        }

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                attentionMask: layer.isMamba ? .none : attentionMask,
                ssmMask: layer.isMamba ? ssmMask : nil,
                cache: cacheValue(in: cacheArray, at: index)
            )
        }
        return hiddenStates
    }

    private func cacheValue(in cache: [KVCache?], at index: Int) -> KVCache? {
        cache.indices.contains(index) ? cache[index] : nil
    }
}

private final class Plamo2Backbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var decoder: Plamo2Decoder
    @ModuleInfo(key: "norm") private var norm: Plamo2RMSNorm

    init(_ config: Plamo2Configuration) {
        precondition(config.vocabularySize > 0, "Plamo2 vocab_size must be positive")
        _embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _decoder.wrappedValue = Plamo2Decoder(config)
        _norm.wrappedValue = Plamo2RMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        norm(decoder(embeddings(inputs), cache: cache))
    }
}

internal final class Plamo2Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let config: Plamo2Configuration
    private let layerPlan: Plamo2LayerPlan

    @ModuleInfo(key: "model") private var model: Plamo2Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: Plamo2Configuration) {
        self.config = config
        self.layerPlan = Plamo2LayerPlan(config)
        self.vocabularySize = config.vocabularySize
        self.kvHeads = layerPlan.kinds.map { $0 == .attention ? config.keyValueHeads : 0 }
        _model.wrappedValue = Plamo2Backbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
        }
        super.init()
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        layerPlan.kinds.map { kind in
            switch kind {
            case .mamba:
                MambaCache()
            case .attention:
                KVCacheSimple()
            }
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(
            model(input[text: .newAxis].tokens, cache: cache)
        )
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        for (key, value) in sanitized where key.contains("conv1d.weight") && value.dim(-1) != 1 {
            sanitized[key] = value.movedAxis(source: 2, destination: 1)
        }

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }
        return sanitized
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) }
            ?? model.embeddings.asLinear(hiddenStates)
    }
}

extension Plamo2Model: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        model.decoder.layers.map { layer in
            switch layer.layerKind {
            case .mamba:
                return (layer.mixer, ["in_proj", "bcdt_proj", "dt_proj", "out_proj"])
            case .attention:
                return (layer.mixer, ["qkv_proj", "o_proj"])
            }
        }
    }
}
