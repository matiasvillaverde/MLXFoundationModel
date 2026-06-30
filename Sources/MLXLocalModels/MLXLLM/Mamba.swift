import Foundation
import MLX
import MLXNN

internal struct MambaConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var vocabularySize: Int
    internal var hiddenSize: Int
    internal var intermediateSize: Int
    internal var stateSize: Int
    internal var hiddenLayers: Int
    internal var convKernel: Int
    internal var useBias: Bool
    internal var useConvBias: Bool
    internal var timeStepRank: Int
    internal var tieWordEmbeddings: Bool
    internal var layerNormEpsilon: Float

    internal init(
        modelType: String = "mamba",
        vocabularySize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        stateSize: Int,
        hiddenLayers: Int,
        convKernel: Int,
        useBias: Bool = false,
        useConvBias: Bool = true,
        timeStepRank: Int? = nil,
        tieWordEmbeddings: Bool = true,
        layerNormEpsilon: Float = 1e-5
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.stateSize = stateSize
        self.hiddenLayers = hiddenLayers
        self.convKernel = convKernel
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.timeStepRank = timeStepRank ?? Self.autoTimeStepRank(hiddenSize: hiddenSize)
        self.tieWordEmbeddings = tieWordEmbeddings
        self.layerNormEpsilon = layerNormEpsilon
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case dModel = "d_model"
        case intermediateSize = "intermediate_size"
        case dInner = "d_inner"
        case stateSize = "state_size"
        case dState = "d_state"
        case hiddenLayers = "num_hidden_layers"
        case nLayer = "n_layer"
        case nLayers = "n_layers"
        case convKernel = "conv_kernel"
        case dConv = "d_conv"
        case useBias = "use_bias"
        case bias
        case useConvBias = "use_conv_bias"
        case convBias = "conv_bias"
        case timeStepRank = "time_step_rank"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerNormEpsilon = "layer_norm_epsilon"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenSize = try Self.decodeRequiredInt(
            from: container,
            primary: .hiddenSize,
            fallback: .dModel
        )
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "mamba",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: hiddenSize,
            intermediateSize: try Self.decodeRequiredInt(
                from: container,
                primary: .intermediateSize,
                fallback: .dInner
            ),
            stateSize: try Self.decodeRequiredInt(
                from: container,
                primary: .stateSize,
                fallback: .dState
            ),
            hiddenLayers: try Self.decodeRequiredInt(
                from: container,
                primary: .hiddenLayers,
                fallbacks: [.nLayer, .nLayers]
            ),
            convKernel: try Self.decodeRequiredInt(
                from: container,
                primary: .convKernel,
                fallback: .dConv
            ),
            useBias: try Self.decodeBool(
                from: container,
                primary: .useBias,
                fallback: .bias,
                default: false
            ),
            useConvBias: try Self.decodeBool(
                from: container,
                primary: .useConvBias,
                fallback: .convBias,
                default: true
            ),
            timeStepRank: try Self.decodeTimeStepRank(from: container, hiddenSize: hiddenSize),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            layerNormEpsilon: try container.decodeIfPresent(
                Float.self,
                forKey: .layerNormEpsilon
            ) ?? 1e-5
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(stateSize, forKey: .stateSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(convKernel, forKey: .convKernel)
        try container.encode(useBias, forKey: .useBias)
        try container.encode(useConvBias, forKey: .useConvBias)
        try container.encode(timeStepRank, forKey: .timeStepRank)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(layerNormEpsilon, forKey: .layerNormEpsilon)
    }

    private static func autoTimeStepRank(hiddenSize: Int) -> Int {
        Int((Double(hiddenSize) / 16.0).rounded(.up))
    }

    private static func decodeRequiredInt(
        from container: KeyedDecodingContainer<CodingKeys>,
        primary: CodingKeys,
        fallback: CodingKeys
    ) throws -> Int {
        try decodeRequiredInt(from: container, primary: primary, fallbacks: [fallback])
    }

    private static func decodeRequiredInt(
        from container: KeyedDecodingContainer<CodingKeys>,
        primary: CodingKeys,
        fallbacks: [CodingKeys]
    ) throws -> Int {
        if let value = try container.decodeIfPresent(Int.self, forKey: primary) {
            return value
        }
        for key in fallbacks {
            if let value = try container.decodeIfPresent(Int.self, forKey: key) {
                return value
            }
        }
        throw DecodingError.keyNotFound(
            primary,
            DecodingError.Context(
                codingPath: container.codingPath,
                debugDescription: "Missing \(primary.stringValue)"
            )
        )
    }

    private static func decodeBool(
        from container: KeyedDecodingContainer<CodingKeys>,
        primary: CodingKeys,
        fallback: CodingKeys,
        default defaultValue: Bool
    ) throws -> Bool {
        try container.decodeIfPresent(Bool.self, forKey: primary)
            ?? container.decodeIfPresent(Bool.self, forKey: fallback)
            ?? defaultValue
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
}

internal struct MambaMixerLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let intermediateSize: Int
    internal let stateSize: Int
    internal let convKernel: Int
    internal let timeStepRank: Int
    internal let inputProjectionSize: Int
    internal let stateProjectionSize: Int

    internal init(_ config: MambaConfiguration) {
        precondition(config.hiddenSize > 0, "Mamba hidden_size must be positive")
        precondition(config.intermediateSize > 0, "Mamba intermediate_size must be positive")
        precondition(config.stateSize > 0, "Mamba state_size must be positive")
        precondition(config.convKernel > 1, "Mamba conv_kernel must be greater than one")
        precondition(config.timeStepRank > 0, "Mamba time_step_rank must be positive")

        self.hiddenSize = config.hiddenSize
        self.intermediateSize = config.intermediateSize
        self.stateSize = config.stateSize
        self.convKernel = config.convKernel
        self.timeStepRank = config.timeStepRank
        self.inputProjectionSize = config.intermediateSize * 2
        self.stateProjectionSize = config.timeStepRank + 2 * config.stateSize
    }
}

private final class MambaMixer: Module {
    private let layout: MambaMixerLayout
    private let useBias: Bool
    private let useConvBias: Bool

    @ModuleInfo(key: "in_proj") fileprivate var inputProjection: Linear
    @ModuleInfo(key: "conv1d") private var convolution: Conv1d
    @ModuleInfo(key: "x_proj") fileprivate var stateProjection: Linear
    @ModuleInfo(key: "dt_proj") fileprivate var timeStepProjection: Linear
    @ModuleInfo(key: "out_proj") fileprivate var outputProjection: Linear

    @ParameterInfo(key: "A_log") private var stateLog: MLXArray
    @ParameterInfo(key: "D") private var skip: MLXArray

    init(_ config: MambaConfiguration) {
        let layout = MambaMixerLayout(config)
        self.layout = layout
        self.useBias = config.useBias
        self.useConvBias = config.useConvBias

        _inputProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.inputProjectionSize,
            bias: useBias
        )
        _convolution.wrappedValue = Conv1d(
            inputChannels: layout.intermediateSize,
            outputChannels: layout.intermediateSize,
            kernelSize: layout.convKernel,
            padding: 0,
            groups: layout.intermediateSize,
            bias: useConvBias
        )
        _stateProjection.wrappedValue = Linear(
            layout.intermediateSize,
            layout.stateProjectionSize,
            bias: false
        )
        _timeStepProjection.wrappedValue = Linear(
            layout.timeStepRank,
            layout.intermediateSize,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            layout.intermediateSize,
            layout.hiddenSize,
            bias: useBias
        )

        let stateValues = MLXArray(Array(1 ... layout.stateSize).map(Float.init))
            .reshaped(1, layout.stateSize)
        _stateLog.wrappedValue = log(repeated(stateValues, count: layout.intermediateSize, axis: 0))
        _skip.wrappedValue = MLXArray.ones([layout.intermediateSize])
    }

    func callAsFunction(_ hiddenStates: MLXArray, cache: MambaCache?) -> MLXArray {
        let convState = cache?[0]
        let recurrentState = cache?[1]
        let (output, newConvState, newRecurrentState) = processSequence(
            hiddenStates,
            convState: convState,
            recurrentState: recurrentState
        )

        if let cache {
            cache[0] = newConvState
            cache[1] = newRecurrentState
            cache.offset += hiddenStates.dim(1)
        }

        return output
    }

    private func processSequence(
        _ hiddenStates: MLXArray,
        convState: MLXArray?,
        recurrentState: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let projected = inputProjection(hiddenStates)
        let parts = projected.split(parts: 2, axis: -1)
        var convInput = parts[0]
        let gate = parts[1]
        let fullConvInput = paddedConvolutionInput(convInput, state: convState)

        let convOutput = convolution(fullConvInput)
        let newConvState = fullConvInput[0..., (1 - layout.convKernel)..., 0...]
        convInput = silu(convOutput)

        let dynamics = -exp(stateLog)
        let (stateOutput, newRecurrentState) = stateSpaceStep(
            convInput,
            dynamics: dynamics,
            state: recurrentState
        )
        return (
            outputProjection(silu(gate) * stateOutput),
            newConvState,
            newRecurrentState
        )
    }

    private func paddedConvolutionInput(_ input: MLXArray, state: MLXArray?) -> MLXArray {
        if let state {
            return concatenated([state, input], axis: 1)
        }
        return padded(
            input,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((layout.convKernel - 1, 0)),
                IntOrPair((0, 0))
            ]
        )
    }

    private func stateSpaceStep(
        _ hiddenStates: MLXArray,
        dynamics: MLXArray,
        state: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let tokenCount = hiddenStates.dim(1)
        let projected = stateProjection(hiddenStates)
        let parts = MLX.split(
            projected,
            indices: [layout.timeStepRank, layout.timeStepRank + layout.stateSize],
            axis: -1
        )
        let timeStep = softplus(timeStepProjection(parts[0]))
        let inputWeights = parts[1]
        let outputWeights = parts[2]

        let stateTrace = expandedDimensions(timeStep * hiddenStates, axis: -1)
            * expandedDimensions(inputWeights, axis: -2)
        let decay = exp(expandedDimensions(timeStep, axis: -1) * dynamics)

        var runningState = state
        for index in 0 ..< tokenCount {
            if let previousState = runningState {
                stateTrace[0..., index] = previousState * decay[0..., index]
                    + stateTrace[0..., index]
            }
            runningState = stateTrace[0..., index]
        }

        let output = stateTrace.matmul(expandedDimensions(outputWeights, axis: -1))
            .squeezed(axis: -1)
        return (output + skip * hiddenStates, stateTrace[0..., -1])
    }
}

private final class MambaBlock: Module {
    @ModuleInfo(key: "mixer") fileprivate var mixer: MambaMixer
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: MambaConfiguration) {
        _mixer.wrappedValue = MambaMixer(config)
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray, cache: MambaCache?) -> MLXArray {
        hiddenStates + mixer(norm(hiddenStates), cache: cache)
    }
}

private final class MambaBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [MambaBlock]
    @ModuleInfo(key: "norm_f") private var finalNorm: RMSNorm

    init(_ config: MambaConfiguration) {
        precondition(config.vocabularySize > 0, "Mamba vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "Mamba num_hidden_layers must be positive")

        _embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            MambaBlock(config)
        }
        _finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embeddings(inputs)
        let cacheArray: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, cache: cacheArray[index] as? MambaCache)
        }
        return finalNorm(hiddenStates)
    }
}

internal final class MambaModel: Module, LLMModel, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let configuration: MambaConfiguration

    @ModuleInfo(key: "backbone") private var backbone: MambaBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: MambaConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _backbone.wrappedValue = MambaBackbone(configuration)
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
        greedyTokenOutput(
            logits: logits(from: lastTokenHiddenState(backbone(input[text: .newAxis].tokens, cache: cache))),
            state: state
        )
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

extension MambaModel {
    func loraLinearLayers() -> LoRALinearLayers {
        backbone.layers.map { layer in
            (layer.mixer, ["in_proj", "x_proj", "dt_proj", "out_proj"])
        }
    }
}
