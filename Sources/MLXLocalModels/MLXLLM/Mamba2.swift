import Foundation
import MLX
import MLXFast
import MLXNN

internal struct Mamba2Configuration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var numHeads: Int
    internal var headDim: Int
    internal var vocabularySize: Int
    internal var hiddenSize: Int
    internal var stateSize: Int
    internal var hiddenLayers: Int
    internal var layerNormEpsilon: Float
    internal var convKernel: Int
    internal var groups: Int
    internal var useBias: Bool
    internal var useConvBias: Bool
    internal var tieWordEmbeddings: Bool
    internal var timeStepLimitMin: Float
    internal var timeStepLimitMax: Float
    internal var timeStepRank: Int

    internal var intermediateSize: Int {
        numHeads * headDim
    }

    internal init(
        modelType: String = "mamba2",
        numHeads: Int,
        headDim: Int,
        vocabularySize: Int,
        hiddenSize: Int,
        stateSize: Int,
        hiddenLayers: Int,
        layerNormEpsilon: Float = 1e-5,
        convKernel: Int = 4,
        groups: Int = 1,
        useBias: Bool = false,
        useConvBias: Bool = true,
        tieWordEmbeddings: Bool = true,
        timeStepLimitMin: Float = 0.001,
        timeStepLimitMax: Float = 100,
        timeStepRank: Int? = nil
    ) {
        self.modelType = modelType
        self.numHeads = numHeads
        self.headDim = headDim
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.stateSize = stateSize
        self.hiddenLayers = hiddenLayers
        self.layerNormEpsilon = layerNormEpsilon
        self.convKernel = convKernel
        self.groups = groups
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
        self.timeStepRank = timeStepRank ?? Self.autoTimeStepRank(hiddenSize: hiddenSize)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case stateSize = "state_size"
        case ssmStateSize = "ssm_state_size"
        case hiddenLayers = "num_hidden_layers"
        case layerNormEpsilon = "layer_norm_epsilon"
        case convKernel = "conv_kernel"
        case groups = "n_groups"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case timeStepLimit = "time_step_limit"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
        case timeStepRank = "time_step_rank"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let limits = try Self.decodeTimeStepLimits(from: container)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "mamba2",
            numHeads: try container.decode(Int.self, forKey: .numHeads),
            headDim: try container.decode(Int.self, forKey: .headDim),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: hiddenSize,
            stateSize: try container.decodeIfPresent(Int.self, forKey: .ssmStateSize)
                ?? container.decode(Int.self, forKey: .stateSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            layerNormEpsilon: try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon)
                ?? 1e-5,
            convKernel: try container.decodeIfPresent(Int.self, forKey: .convKernel) ?? 4,
            groups: try container.decodeIfPresent(Int.self, forKey: .groups) ?? 1,
            useBias: try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            useConvBias: try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            timeStepLimitMin: limits.minimum,
            timeStepLimitMax: limits.maximum,
            timeStepRank: try Self.decodeTimeStepRank(from: container, hiddenSize: hiddenSize)
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(numHeads, forKey: .numHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(stateSize, forKey: .stateSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(layerNormEpsilon, forKey: .layerNormEpsilon)
        try container.encode(convKernel, forKey: .convKernel)
        try container.encode(groups, forKey: .groups)
        try container.encode(useBias, forKey: .useBias)
        try container.encode(useConvBias, forKey: .useConvBias)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(timeStepLimitMin, forKey: .timeStepMin)
        if timeStepLimitMax.isFinite {
            try container.encode(timeStepLimitMax, forKey: .timeStepMax)
        }
        try container.encode(timeStepRank, forKey: .timeStepRank)
    }

    private static func autoTimeStepRank(hiddenSize: Int) -> Int {
        Int((Double(hiddenSize) / 16.0).rounded(.up))
    }

    private static func decodeTimeStepRank(
        from container: KeyedDecodingContainer<CodingKeys>,
        hiddenSize: Int
    ) throws -> Int {
        if let value = try? container.decodeIfPresent(Int.self, forKey: .timeStepRank) {
            return value
        }
        guard let value = try container.decodeIfPresent(String.self, forKey: .timeStepRank) else {
            return autoTimeStepRank(hiddenSize: hiddenSize)
        }
        if value == "auto" {
            return autoTimeStepRank(hiddenSize: hiddenSize)
        }
        if let intValue = Int(value) {
            return intValue
        }
        throw DecodingError.dataCorruptedError(
            forKey: .timeStepRank,
            in: container,
            debugDescription: "time_step_rank must be an integer or \"auto\""
        )
    }

    private static func decodeTimeStepLimits(
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> (minimum: Float, maximum: Float) {
        if let limits = try container.decodeIfPresent([Float].self, forKey: .timeStepLimit) {
            guard !limits.isEmpty else {
                throw DecodingError.dataCorruptedError(
                    forKey: .timeStepLimit,
                    in: container,
                    debugDescription: "time_step_limit cannot be empty"
                )
            }
            return (limits[0], limits.count > 1 ? limits[1] : limits[0])
        }
        return (
            try container.decodeIfPresent(Float.self, forKey: .timeStepMin) ?? 0.001,
            try container.decodeIfPresent(Float.self, forKey: .timeStepMax) ?? 100
        )
    }
}

internal struct Mamba2MixerLayout: Sendable {
    internal let hiddenSize: Int
    internal let heads: Int
    internal let headDim: Int
    internal let stateSize: Int
    internal let groups: Int
    internal let convKernel: Int
    internal let intermediateSize: Int
    internal let convInputSize: Int
    internal let inputProjectionSize: Int
    internal let timeStepLimit: (Float, Float)

    internal init(_ config: Mamba2Configuration) {
        precondition(config.hiddenSize > 0, "Mamba2 hidden_size must be positive")
        precondition(config.numHeads > 0, "Mamba2 num_heads must be positive")
        precondition(config.headDim > 0, "Mamba2 head_dim must be positive")
        precondition(config.stateSize > 0, "Mamba2 state_size must be positive")
        precondition(config.groups > 0, "Mamba2 n_groups must be positive")
        precondition(
            config.numHeads.isMultiple(of: config.groups),
            "Mamba2 heads must divide groups"
        )
        precondition(config.convKernel > 1, "Mamba2 conv_kernel must be greater than one")
        precondition(config.timeStepRank > 0, "Mamba2 time_step_rank must be positive")

        self.hiddenSize = config.hiddenSize
        self.heads = config.numHeads
        self.headDim = config.headDim
        self.stateSize = config.stateSize
        self.groups = config.groups
        self.convKernel = config.convKernel
        self.intermediateSize = config.intermediateSize
        self.convInputSize = intermediateSize + 2 * groups * stateSize
        self.inputProjectionSize = intermediateSize + convInputSize + heads
        self.timeStepLimit = (config.timeStepLimitMin, config.timeStepLimitMax)
    }
}

private final class Mamba2RMSNormGated: Module {
    @ParameterInfo(key: "weight") private var weight: MLXArray

    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(hiddenStates * silu(gate), weight: weight, eps: eps)
    }
}

private final class Mamba2Mixer: Module {
    private let layout: Mamba2MixerLayout

    @ModuleInfo(key: "conv1d") private var convolution: Conv1d
    @ModuleInfo(key: "in_proj") fileprivate var inputProjection: Linear
    @ModuleInfo(key: "out_proj") fileprivate var outputProjection: Linear
    @ModuleInfo(key: "norm") private var norm: Mamba2RMSNormGated

    @ParameterInfo(key: "dt_bias") private var timeStepBias: MLXArray
    @ParameterInfo(key: "A_log") private var stateLog: MLXArray
    @ParameterInfo(key: "D") private var residualScale: MLXArray

    init(_ config: Mamba2Configuration) {
        let layout = Mamba2MixerLayout(config)
        self.layout = layout
        _convolution.wrappedValue = Conv1d(
            inputChannels: layout.convInputSize,
            outputChannels: layout.convInputSize,
            kernelSize: layout.convKernel,
            padding: 0,
            groups: layout.convInputSize,
            bias: config.useConvBias
        )
        _inputProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.inputProjectionSize,
            bias: config.useBias
        )
        _outputProjection.wrappedValue = Linear(
            layout.intermediateSize,
            layout.hiddenSize,
            bias: config.useBias
        )
        _norm.wrappedValue = Mamba2RMSNormGated(
            dimensions: layout.intermediateSize,
            eps: config.layerNormEpsilon
        )
        _timeStepBias.wrappedValue = MLXArray.ones([layout.heads])
        _stateLog.wrappedValue = log((MLXArray(0 ..< layout.heads).asType(.float32) + 1))
        _residualScale.wrappedValue = MLXArray.ones([layout.heads])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        let projected = inputProjection(hiddenStates)
        let parts = split(
            projected,
            indices: [
                layout.intermediateSize,
                layout.intermediateSize + layout.convInputSize,
            ],
            axis: -1
        )
        let gate = parts[0]
        var convInput = parts[1]
        let timeSteps = parts[2]

        if let mask {
            convInput = MLX.where(
                expandedDimensions(mask, axis: -1),
                convInput,
                MLXArray.zeros(like: convInput)
            )
        }

        let convOutput = applyConvolution(convInput, cache: cache)
        let convParts = split(
            convOutput,
            indices: [
                layout.intermediateSize,
                layout.intermediateSize + layout.groups * layout.stateSize,
            ],
            axis: -1
        )

        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)
        let ssmInput = convParts[0].reshaped(
            batchSize,
            tokenCount,
            layout.heads,
            layout.headDim
        )
        let inputState = convParts[1].reshaped(
            batchSize,
            tokenCount,
            layout.groups,
            layout.stateSize
        )
        let outputState = convParts[2].reshaped(
            batchSize,
            tokenCount,
            layout.groups,
            layout.stateSize
        )
        let reshapedTimeSteps = timeSteps.reshaped(batchSize, tokenCount, layout.heads)
        let (output, nextState) = ssmUpdate(
            hiddenStates: ssmInput,
            ALog: stateLog,
            B: inputState,
            C: outputState,
            D: residualScale,
            dt: reshapedTimeSteps,
            dtBias: timeStepBias,
            state: cache?[1],
            timeStepLimit: layout.timeStepLimit,
            mask: mask
        )

        if let cache {
            cache[1] = nextState
            cache.offset += tokenCount
        }

        return outputProjection(norm(output.flattened(start: 2), gate: gate))
    }

    private func applyConvolution(_ input: MLXArray, cache: MambaCache?) -> MLXArray {
        let batchSize = input.dim(0)
        let stateLength = layout.convKernel - 1
        let state = cache?[0] ?? MLXArray.zeros(
            [batchSize, stateLength, layout.convInputSize],
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

private final class Mamba2Block: Module {
    @ModuleInfo(key: "mixer") fileprivate var mixer: Mamba2Mixer
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: Mamba2Configuration) {
        _mixer.wrappedValue = Mamba2Mixer(config)
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        hiddenStates + mixer(norm(hiddenStates), mask: mask, cache: cache)
    }
}

private final class Mamba2Backbone: Module {
    @ModuleInfo(key: "embeddings") fileprivate var embeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [Mamba2Block]
    @ModuleInfo(key: "norm_f") private var finalNorm: RMSNorm

    init(_ config: Mamba2Configuration) {
        precondition(config.vocabularySize > 0, "Mamba2 vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "Mamba2 num_hidden_layers must be positive")

        _embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Mamba2Block(config)
        }
        _finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embeddings(inputs)
        let cacheArray: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let mask = createSSMMask(h: hiddenStates, cache: cacheArray.first as? MambaCache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cacheArray[index] as? MambaCache)
        }

        return finalNorm(hiddenStates)
    }
}

internal final class Mamba2Model: Module, LLMModel, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let configuration: Mamba2Configuration

    @ModuleInfo(key: "backbone") private var backbone: Mamba2Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: Mamba2Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _backbone.wrappedValue = Mamba2Backbone(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { _ in MambaCache() }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: backbone(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(
            backbone(input[text: .newAxis].tokens, cache: cache)
        )
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for (key, value) in weights where key.contains("conv1d.weight") && value.dim(-1) != 1 {
            weights[key] = value.movedAxis(source: 2, destination: 1)
        }
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["lm_head.scales"] = nil
            weights["lm_head.biases"] = nil
        }
        return weights
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return backbone.embeddings.asLinear(hiddenStates)
    }
}

extension Mamba2Model {
    func loraLinearLayers() -> LoRALinearLayers {
        backbone.layers.map { layer in
            (layer.mixer, ["in_proj", "out_proj"])
        }
    }
}
