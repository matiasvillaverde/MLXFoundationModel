import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

internal struct RWKV7Configuration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var vocabularySize: Int
    internal var hiddenSize: Int
    internal var intermediateSize: Int
    internal var hiddenLayers: Int
    internal var headSize: Int
    internal var layerNormEpsilon: Float
    internal var groupNormEpsilon: Float
    internal var inContextLearningRank: Int
    internal var valueRank: Int
    internal var gateRank: Int
    internal var decayRank: Int
    internal var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "rwkv7",
        vocabularySize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        hiddenLayers: Int,
        headSize: Int = 64,
        layerNormEpsilon: Float = 1e-5,
        groupNormEpsilon: Float = 64e-5,
        inContextLearningRank: Int = 64,
        valueRank: Int = 32,
        gateRank: Int = 128,
        decayRank: Int = 64,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.headSize = headSize
        self.layerNormEpsilon = layerNormEpsilon
        self.groupNormEpsilon = groupNormEpsilon
        self.inContextLearningRank = inContextLearningRank
        self.valueRank = valueRank
        self.gateRank = gateRank
        self.decayRank = decayRank
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    internal var headCount: Int {
        hiddenSize / headSize
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case headSize = "head_size"
        case headDim = "head_dim"
        case layerNormEpsilon = "layer_norm_epsilon"
        case normEps = "norm_eps"
        case groupNormEpsilon = "group_norm_epsilon"
        case inContextLearningRank = "in_context_learning_lora_rank"
        case aLowRankDim = "a_low_rank_dim"
        case valueRank = "value_lora_rank"
        case vLowRankDim = "v_low_rank_dim"
        case gateRank = "gate_lora_rank"
        case gateLowRankDim = "gate_low_rank_dim"
        case decayRank = "decay_lora_rank"
        case decayLowRankDim = "decay_low_rank_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "rwkv7",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            headSize: try Self.decodeInt(from: container, keys: [.headSize, .headDim], default: 64),
            layerNormEpsilon: try Self.decodeFloat(
                from: container,
                keys: [.layerNormEpsilon, .normEps],
                default: 1e-5
            ),
            groupNormEpsilon: try container.decodeIfPresent(
                Float.self,
                forKey: .groupNormEpsilon
            ) ?? 64e-5,
            inContextLearningRank: try Self.decodeInt(
                from: container,
                keys: [.inContextLearningRank, .aLowRankDim],
                default: 64
            ),
            valueRank: try Self.decodeInt(
                from: container,
                keys: [.valueRank, .vLowRankDim],
                default: 32
            ),
            gateRank: try Self.decodeInt(
                from: container,
                keys: [.gateRank, .gateLowRankDim],
                default: 128
            ),
            decayRank: try Self.decodeInt(
                from: container,
                keys: [.decayRank, .decayLowRankDim],
                default: 64
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(headSize, forKey: .headSize)
        try container.encode(layerNormEpsilon, forKey: .layerNormEpsilon)
        try container.encode(groupNormEpsilon, forKey: .groupNormEpsilon)
        try container.encode(inContextLearningRank, forKey: .inContextLearningRank)
        try container.encode(valueRank, forKey: .valueRank)
        try container.encode(gateRank, forKey: .gateRank)
        try container.encode(decayRank, forKey: .decayRank)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }

    private static func decodeInt(
        from container: KeyedDecodingContainer<CodingKeys>,
        keys: [CodingKeys],
        default defaultValue: Int
    ) throws -> Int {
        for key in keys {
            if let value = try container.decodeIfPresent(Int.self, forKey: key) {
                return value
            }
        }
        return defaultValue
    }

    private static func decodeFloat(
        from container: KeyedDecodingContainer<CodingKeys>,
        keys: [CodingKeys],
        default defaultValue: Float
    ) throws -> Float {
        for key in keys {
            if let value = try container.decodeIfPresent(Float.self, forKey: key) {
                return value
            }
        }
        return defaultValue
    }
}

// MARK: - WKV recurrence

internal struct RWKV7WKVLayout: Equatable, Sendable {
    internal let batchSize: Int
    internal let sequenceLength: Int
    internal let headCount: Int
    internal let headSize: Int

    internal init(receptance: MLXArray, key: MLXArray, value: MLXArray) {
        precondition(receptance.ndim == 4, "RWKV7 receptance must be [batch, time, heads, head]")
        precondition(key.shape == receptance.shape, "RWKV7 key shape must match receptance")
        precondition(value.shape == receptance.shape, "RWKV7 value shape must match receptance")

        self.batchSize = receptance.dim(0)
        self.sequenceLength = receptance.dim(1)
        self.headCount = receptance.dim(2)
        self.headSize = receptance.dim(3)
    }

    internal var stateShape: [Int] {
        [batchSize, headCount, headSize, headSize]
    }

    internal var supportsMetalKernel: Bool {
        headSize > 0 && headSize.isMultiple(of: 32)
    }
}

private func makeRWKV7WKVKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            auto sampleAndHead = thread_position_in_grid.z;
            auto batchIndex = sampleAndHead / headCount;
            auto headIndex = sampleAndHead % headCount;
            constexpr int piecesPerThread = headSize / 32;

            auto rToken = r + batchIndex * sequenceLength * headCount * headSize
                + headIndex * headSize;
            auto wToken = w + batchIndex * sequenceLength * headCount * headSize
                + headIndex * headSize;
            auto kToken = k + batchIndex * sequenceLength * headCount * headSize
                + headIndex * headSize;
            auto vToken = v + batchIndex * sequenceLength * headCount * headSize
                + headIndex * headSize;
            auto aToken = a + batchIndex * sequenceLength * headCount * headSize
                + headIndex * headSize;
            auto bToken = b + batchIndex * sequenceLength * headCount * headSize
                + headIndex * headSize;
            auto yToken = y + batchIndex * sequenceLength * headCount * headSize
                + headIndex * headSize;

            auto keyLane = thread_position_in_threadgroup.x;
            auto valueChannel = thread_position_in_grid.y;
            auto stateOffset = (sampleAndHead * headSize + valueChannel) * headSize;
            auto stateRead = stateIn + stateOffset;
            auto stateWrite = stateOut + stateOffset;

            float laneState[piecesPerThread];
            for (int piece = 0; piece < piecesPerThread; ++piece) {
                auto keyOffset = piecesPerThread * keyLane + piece;
                laneState[piece] = static_cast<float>(stateRead[keyOffset]);
            }

            for (int tokenIndex = 0; tokenIndex < sequenceLength; ++tokenIndex) {
                float stateByA = 0.0f;
                for (int piece = 0; piece < piecesPerThread; ++piece) {
                    auto keyOffset = piecesPerThread * keyLane + piece;
                    stateByA += laneState[piece] * static_cast<float>(aToken[keyOffset]);
                    laneState[piece] *= static_cast<float>(wToken[keyOffset]);
                }
                stateByA = simd_sum(stateByA);

                float output = 0.0f;
                for (int piece = 0; piece < piecesPerThread; ++piece) {
                    auto keyOffset = piecesPerThread * keyLane + piece;
                    laneState[piece] += static_cast<float>(vToken[valueChannel])
                        * static_cast<float>(kToken[keyOffset])
                        + stateByA * static_cast<float>(bToken[keyOffset]);
                    output += laneState[piece] * static_cast<float>(rToken[keyOffset]);
                }
                output = simd_sum(output);

                if (thread_index_in_simdgroup == 0) {
                    yToken[valueChannel] = static_cast<InT>(output);
                }

                rToken += headCount * headSize;
                wToken += headCount * headSize;
                kToken += headCount * headSize;
                vToken += headCount * headSize;
                aToken += headCount * headSize;
                bToken += headCount * headSize;
                yToken += headCount * headSize;
            }

            for (int piece = 0; piece < piecesPerThread; ++piece) {
                auto keyOffset = piecesPerThread * keyLane + piece;
                stateWrite[keyOffset] = static_cast<InT>(laneState[piece]);
            }
        """

    return MLXFast.metalKernel(
        name: "rwkv7_wkv_recurrence",
        inputNames: ["r", "w", "k", "v", "a", "b", "stateIn", "sequenceLength"],
        outputNames: ["y", "stateOut"],
        source: source
    )
}

private final class RWKV7WKVKernelPool: Sendable {
    static let shared = RWKV7WKVKernelPool()

    let kernel: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeRWKV7WKVKernel()
    }
}

private func rwkv7WKVKernel(
    receptance: MLXArray,
    decay: MLXArray,
    key: MLXArray,
    value: MLXArray,
    stateKey: MLXArray,
    stateValue: MLXArray,
    state: MLXArray
) -> (MLXArray, MLXArray) {
    let layout = RWKV7WKVLayout(receptance: receptance, key: key, value: value)
    guard layout.supportsMetalKernel, let kernel = RWKV7WKVKernelPool.shared.kernel else {
        return rwkv7WKVOps(
            receptance: receptance,
            decay: decay,
            key: key,
            value: value,
            stateKey: stateKey,
            stateValue: stateValue,
            state: state
        )
    }

    let outputs = kernel(
        [receptance, decay, key, value, stateKey, stateValue, state, MLXArray(layout.sequenceLength)],
        template: [
            ("InT", receptance.dtype),
            ("headSize", layout.headSize),
            ("headCount", layout.headCount),
        ],
        grid: (32, layout.headSize, layout.batchSize * layout.headCount),
        threadGroup: (32, 4, 1),
        outputShapes: [[
            layout.batchSize,
            layout.sequenceLength,
            layout.headCount,
            layout.headSize,
        ], state.shape],
        outputDTypes: [receptance.dtype, receptance.dtype]
    )

    return (outputs[0], outputs[1])
}

private func rwkv7WKVStepOps(
    receptance: MLXArray,
    decay: MLXArray,
    key: MLXArray,
    value: MLXArray,
    stateKey: MLXArray,
    stateValue: MLXArray,
    state: MLXArray
) -> (MLXArray, MLXArray) {
    let stateByKey = state.matmul(expandedDimensions(stateKey, axis: -1))
        .matmul(expandedDimensions(stateValue, axis: -2))
    let valueByKey = expandedDimensions(value, axis: -1)
        .matmul(expandedDimensions(key, axis: -2))
    let nextState = state * expandedDimensions(decay, axis: -2) + stateByKey + valueByKey
    let output = nextState.matmul(expandedDimensions(receptance, axis: -1)).squeezed(axis: -1)
    return (output, nextState)
}

private func rwkv7WKVOps(
    receptance: MLXArray,
    decay: MLXArray,
    key: MLXArray,
    value: MLXArray,
    stateKey: MLXArray,
    stateValue: MLXArray,
    state: MLXArray
) -> (MLXArray, MLXArray) {
    let layout = RWKV7WKVLayout(receptance: receptance, key: key, value: value)
    var state = state
    var outputs = [MLXArray]()
    outputs.reserveCapacity(layout.sequenceLength)

    for tokenIndex in 0 ..< layout.sequenceLength {
        let (output, nextState) = rwkv7WKVStepOps(
            receptance: receptance[0..., tokenIndex],
            decay: decay[0..., tokenIndex],
            key: key[0..., tokenIndex],
            value: value[0..., tokenIndex],
            stateKey: stateKey[0..., tokenIndex],
            stateValue: stateValue[0..., tokenIndex],
            state: state
        )
        outputs.append(output)
        state = nextState
    }

    return (MLX.stacked(outputs, axis: 1), state)
}

internal func rwkv7WKVUpdate(
    receptance: MLXArray,
    decay: MLXArray,
    key: MLXArray,
    value: MLXArray,
    stateKey: MLXArray,
    stateValue: MLXArray,
    state: MLXArray?
) -> (MLXArray, MLXArray) {
    let layout = RWKV7WKVLayout(receptance: receptance, key: key, value: value)
    let initialState = state ?? MLXArray.zeros(layout.stateShape, dtype: receptance.dtype)
    guard layout.supportsMetalKernel else {
        return rwkv7WKVOps(
            receptance: receptance,
            decay: decay,
            key: key,
            value: value,
            stateKey: stateKey,
            stateValue: stateValue,
            state: initialState
        )
    }

    return rwkv7WKVKernel(
        receptance: receptance,
        decay: decay,
        key: key,
        value: value,
        stateKey: stateKey,
        stateValue: stateValue,
        state: initialState
    )
}

// MARK: - Layers

private func rwkv7Lerp(_ current: MLXArray, previous: MLXArray, weight: MLXArray) -> MLXArray {
    current + weight * (previous - current)
}

private func rwkv7L2Normalize(_ input: MLXArray) -> MLXArray {
    input / maximum(sqrt(sum(input * input, axes: [-1], keepDims: true)), MLXArray(1e-12))
}

private final class RWKV7TokenShift {
    func callAsFunction(_ input: MLXArray, state: MLXArray?) -> MLXArray {
        let batchSize = input.dim(0)
        let hiddenSize = input.dim(2)
        let previous = state ?? MLXArray.zeros([batchSize, 1, hiddenSize], dtype: input.dtype)
        guard input.dim(1) > 1 else {
            return previous
        }
        return concatenated([previous, input[0..., ..<(-1), 0...]], axis: 1)
    }
}

private final class RWKV7LayerNormPerHead: Module {
    @ParameterInfo(key: "weight") private var weight: MLXArray
    @ParameterInfo(key: "bias") private var bias: MLXArray

    private let hiddenSize: Int
    private let eps: Float

    init(hiddenSize: Int, eps: Float) {
        self.hiddenSize = hiddenSize
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([hiddenSize])
        _bias.wrappedValue = MLXArray.zeros([hiddenSize])
        super.init()
    }

    func callAsFunction(_ states: MLXArray) -> MLXArray {
        let normalized = MLXFast.layerNorm(states, weight: nil, bias: nil, eps: eps)
        return normalized.reshaped(states.dim(0), states.dim(1), hiddenSize) * weight + bias
    }
}

private final class RWKV7ChannelMixing: Module {
    private let tokenShift = RWKV7TokenShift()

    @ParameterInfo(key: "x_k") private var keyMix: MLXArray
    @ModuleInfo(key: "key") private var keyProjection: Linear
    @ModuleInfo(key: "value") private var valueProjection: Linear

    init(_ config: RWKV7Configuration) {
        _keyMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ normalizedInput: MLXArray, residual: MLXArray, cache: MambaCache?) -> MLXArray {
        let previous = tokenShift(normalizedInput, state: cache?[0])
        let mixed = rwkv7Lerp(normalizedInput, previous: previous, weight: keyMix)
        let activated = relu(keyProjection(mixed))
        if let cache {
            cache[0] = normalizedInput[0..., (normalizedInput.dim(1) - 1) ..< normalizedInput.dim(1), 0...]
            cache.offset += normalizedInput.dim(1)
        }
        return residual + valueProjection(activated * activated)
    }
}

private final class RWKV7TimeMixing: Module {
    private let tokenShift = RWKV7TokenShift()
    private let headCount: Int
    private let headSize: Int

    @ParameterInfo(key: "x_r") private var receptanceMix: MLXArray
    @ParameterInfo(key: "x_w") private var decayMix: MLXArray
    @ParameterInfo(key: "x_k") private var keyMix: MLXArray
    @ParameterInfo(key: "x_v") private var valueMix: MLXArray
    @ParameterInfo(key: "x_a") private var stateMix: MLXArray
    @ParameterInfo(key: "x_g") private var gateMix: MLXArray

    @ParameterInfo(key: "w0") private var decayBase: MLXArray
    @ParameterInfo(key: "w1") private var decayDown: MLXArray
    @ParameterInfo(key: "w2") private var decayUp: MLXArray
    @ParameterInfo(key: "a0") private var stateBase: MLXArray
    @ParameterInfo(key: "a1") private var stateDown: MLXArray
    @ParameterInfo(key: "a2") private var stateUp: MLXArray
    @ParameterInfo(key: "v0") private var valueBase: MLXArray
    @ParameterInfo(key: "v1") private var valueDown: MLXArray
    @ParameterInfo(key: "v2") private var valueUp: MLXArray
    @ParameterInfo(key: "g1") private var gateDown: MLXArray
    @ParameterInfo(key: "g2") private var gateUp: MLXArray
    @ParameterInfo(key: "k_k") private var keyNormalizer: MLXArray
    @ParameterInfo(key: "k_a") private var keyStateMix: MLXArray
    @ParameterInfo(key: "r_k") private var receptanceKeyGate: MLXArray

    @ModuleInfo(key: "receptance") private var receptanceProjection: Linear
    @ModuleInfo(key: "key") private var keyProjection: Linear
    @ModuleInfo(key: "value") private var valueProjection: Linear
    @ModuleInfo(key: "output") private var outputProjection: Linear
    @ModuleInfo(key: "ln_x") private var outputNorm: RWKV7LayerNormPerHead

    init(_ config: RWKV7Configuration) {
        self.headCount = config.headCount
        self.headSize = config.headSize

        _receptanceMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _decayMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _keyMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _valueMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _stateMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _gateMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _decayBase.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _decayDown.wrappedValue = MLXArray.zeros([config.hiddenSize, config.decayRank])
        _decayUp.wrappedValue = MLXArray.zeros([config.decayRank, config.hiddenSize])
        _stateBase.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _stateDown.wrappedValue = MLXArray.zeros([config.hiddenSize, config.inContextLearningRank])
        _stateUp.wrappedValue = MLXArray.zeros([config.inContextLearningRank, config.hiddenSize])
        _valueBase.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _valueDown.wrappedValue = MLXArray.zeros([config.hiddenSize, config.valueRank])
        _valueUp.wrappedValue = MLXArray.zeros([config.valueRank, config.hiddenSize])
        _gateDown.wrappedValue = MLXArray.zeros([config.hiddenSize, config.gateRank])
        _gateUp.wrappedValue = MLXArray.zeros([config.gateRank, config.hiddenSize])
        _keyNormalizer.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _keyStateMix.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _receptanceKeyGate.wrappedValue = MLXArray.zeros([headCount, headSize])

        _receptanceProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _outputProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _outputNorm.wrappedValue = RWKV7LayerNormPerHead(
            hiddenSize: config.hiddenSize,
            eps: config.groupNormEpsilon
        )
        super.init()
    }

    func callAsFunction(
        _ normalizedInput: MLXArray,
        residual: MLXArray,
        valueFirst: MLXArray?,
        cache: MambaCache?
    ) -> (MLXArray, MLXArray) {
        let batchSize = normalizedInput.dim(0)
        let tokenCount = normalizedInput.dim(1)
        let previous = tokenShift(normalizedInput, state: cache?[0])

        let receptanceInput = rwkv7Lerp(
            normalizedInput,
            previous: previous,
            weight: receptanceMix
        )
        let decayInput = rwkv7Lerp(normalizedInput, previous: previous, weight: decayMix)
        let keyInput = rwkv7Lerp(normalizedInput, previous: previous, weight: keyMix)
        let valueInput = rwkv7Lerp(normalizedInput, previous: previous, weight: valueMix)
        let stateInput = rwkv7Lerp(normalizedInput, previous: previous, weight: stateMix)
        let gateInput = rwkv7Lerp(normalizedInput, previous: previous, weight: gateMix)

        var value = valueProjection(valueInput)
        let firstValue = valueFirst ?? value
        if valueFirst != nil {
            let valueGate = sigmoid(valueBase + valueInput.matmul(valueDown).matmul(valueUp))
            value = rwkv7Lerp(value, previous: firstValue, weight: valueGate)
        }

        let decay = exp(
            -sigmoid(decayBase + tanh(decayInput.matmul(decayDown)).matmul(decayUp))
                * MLXArray(0.60653067)
        )
        let stateGate = sigmoid(stateBase + stateInput.matmul(stateDown).matmul(stateUp))
        let gate = sigmoid(gateInput.matmul(gateDown)).matmul(gateUp)

        let receptance = receptanceProjection(receptanceInput)
            .reshaped(batchSize, tokenCount, headCount, headSize)
        var key = keyProjection(keyInput)
        let normalizedKey = rwkv7L2Normalize(
            (key * keyNormalizer).reshaped(batchSize, tokenCount, headCount, headSize)
        )
        key = rwkv7Lerp(key, previous: key * stateGate, weight: keyStateMix)

        let keyHeads = key.reshaped(batchSize, tokenCount, headCount, headSize)
        let valueHeads = value.reshaped(batchSize, tokenCount, headCount, headSize)
        let decayHeads = decay.reshaped(batchSize, tokenCount, headCount, headSize)
        let stateValue = -normalizedKey * stateGate.reshaped(
            batchSize,
            tokenCount,
            headCount,
            headSize
        )

        let (weighted, newState) = rwkv7WKVUpdate(
            receptance: receptance,
            decay: decayHeads,
            key: keyHeads,
            value: valueHeads,
            stateKey: normalizedKey,
            stateValue: stateValue,
            state: cache?[1]
        )

        let residualGate = (
            receptance * keyHeads * receptanceKeyGate
        ).sum(axis: -1, keepDims: true) * valueHeads
        let mixed = outputNorm(weighted) + residualGate.reshaped(
            batchSize,
            tokenCount,
            headCount * headSize
        )

        if let cache {
            cache[0] = normalizedInput[0..., (tokenCount - 1) ..< tokenCount, 0...]
            cache[1] = newState
            cache.offset += tokenCount
        }

        return (residual + outputProjection(mixed * gate), firstValue)
    }
}

private final class RWKV7Block: Module {
    @ModuleInfo(key: "pre_ln") private var preLayerNorm: LayerNorm?
    @ModuleInfo(key: "ln1") private var attentionLayerNorm: LayerNorm
    @ModuleInfo(key: "attention") fileprivate var attention: RWKV7TimeMixing
    @ModuleInfo(key: "ln2") private var feedForwardLayerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") fileprivate var feedForward: RWKV7ChannelMixing

    init(_ config: RWKV7Configuration, layerIndex: Int) {
        _preLayerNorm.wrappedValue = layerIndex == 0
            ? LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
            : nil
        _attentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _attention.wrappedValue = RWKV7TimeMixing(config)
        _feedForwardLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _feedForward.wrappedValue = RWKV7ChannelMixing(config)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        valueFirst: MLXArray?,
        cache: CacheList?
    ) -> (MLXArray, MLXArray) {
        let normalizedInput = preLayerNorm.map { $0(input) } ?? input
        let (attended, nextValueFirst) = attention(
            attentionLayerNorm(normalizedInput),
            residual: normalizedInput,
            valueFirst: valueFirst,
            cache: cache?[0] as? MambaCache
        )
        let output = feedForward(
            feedForwardLayerNorm(attended),
            residual: attended,
            cache: cache?[1] as? MambaCache
        )
        return (output, nextValueFirst)
    }
}

private final class RWKV7Backbone: Module {
    @ModuleInfo(key: "embeddings") fileprivate var embeddings: Embedding
    @ModuleInfo(key: "blocks") fileprivate var blocks: [RWKV7Block]
    @ModuleInfo(key: "ln_out") private var outputLayerNorm: LayerNorm

    init(_ config: RWKV7Configuration) {
        precondition(config.vocabularySize > 0, "RWKV7 vocab_size must be positive")
        precondition(config.hiddenSize > 0, "RWKV7 hidden_size must be positive")
        precondition(config.hiddenLayers > 0, "RWKV7 num_hidden_layers must be positive")
        precondition(config.headSize > 0, "RWKV7 head_size must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.headSize),
            "RWKV7 hidden_size must be divisible by head_size"
        )

        _embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _blocks.wrappedValue = (0 ..< config.hiddenLayers).map {
            RWKV7Block(config, layerIndex: $0)
        }
        _outputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embeddings(tokens)
        let cache = cache as? [CacheList]
        var valueFirst: MLXArray?
        for (index, block) in blocks.enumerated() {
            (hiddenStates, valueFirst) = block(
                hiddenStates,
                valueFirst: valueFirst,
                cache: cache?[index]
            )
        }
        return outputLayerNorm(hiddenStates)
    }
}

internal final class RWKV7Model: Module, LLMModel, GreedyTokenModel {
    @ModuleInfo(key: "rwkv7") private var backbone: RWKV7Backbone
    @ModuleInfo(key: "head") private var head: Linear?

    private let config: RWKV7Configuration

    internal init(_ config: RWKV7Configuration) {
        self.config = config
        _backbone.wrappedValue = RWKV7Backbone(config)
        _head.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: backbone(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        greedyTokenOutput(
            logits: logits(
                from: lastTokenHiddenState(
                    backbone(input[text: .newAxis].tokens, cache: cache)
                )
            ),
            state: state
        )
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { _ in
            CacheList(MambaCache(), MambaCache())
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let head {
            return head(hiddenStates)
        }
        return backbone.embeddings.asLinear(hiddenStates)
    }
}

