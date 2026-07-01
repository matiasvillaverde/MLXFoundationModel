import Foundation
import MLX
import MLXFast
import MLXNN

internal struct IQuestLoopCoderConfiguration: Codable, Equatable, Sendable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var rmsNormEps: Float
    internal var vocabularySize: Int
    internal var headDim: Int
    internal var keyValueHeads: Int
    internal var maxPositionEmbeddings: Int
    internal var attentionBias: Bool
    internal var mlpBias: Bool
    internal var ropeTheta: Float
    internal var ropeScaling: [String: StringOrNumber]?
    internal var tieWordEmbeddings: Bool
    internal var loopCount: Int
    internal var loopWindowSize: Int

    internal init(
        modelType: String = "iquestloopcoder",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int,
        headDim: Int,
        keyValueHeads: Int? = nil,
        maxPositionEmbeddings: Int = 131_072,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        ropeTheta: Float = 500_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        loopCount: Int = 2,
        loopWindowSize: Int = 64
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.headDim = headDim
        self.keyValueHeads = keyValueHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.loopCount = loopCount
        self.loopWindowSize = loopWindowSize
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "iquestloopcoder",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-6,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            headDim: try container.decode(Int.self, forKey: .headDim),
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias)
                ?? false,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 500_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            loopCount: try container.decodeIfPresent(Int.self, forKey: .loopCount) ?? 2,
            loopWindowSize: try container.decodeIfPresent(Int.self, forKey: .loopWindowSize) ?? 64
        )
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDim = "head_dim"
        case keyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case loopCount = "loop_num"
        case loopWindowSize = "loop_window_size"
    }
}

private final class IQuestLoopGateProjection: Module {
    @ParameterInfo(key: "weight") private var weight: MLXArray
    @ParameterInfo(key: "bias") private var bias: MLXArray

    init(headCount: Int, headDim: Int) {
        self._weight.wrappedValue = zeros([headCount, headDim])
        self._bias.wrappedValue = zeros([headCount])
    }

    func callAsFunction(_ query: MLXArray) -> MLXArray {
        let projected = query.matmul(
            expandedDimensions(weight, axis: 1).swappedAxes(-1, -2)
        )
        let headBias = expandedDimensions(bias, axes: [0, 2, 3])
        return sigmoid(projected + headBias)
    }
}

private final class IQuestLoopCoderAttention: Module {
    private let attentionHeads: Int
    private let keyValueHeads: Int
    private let headDim: Int
    private let scale: Float
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: IQuestLoopCoderConfiguration) {
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.headDim = config.headDim
        self.scale = Float(1.0 / Double(config.headDim).squareRoot())
        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        self._queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * config.headDim,
            bias: config.attentionBias
        )
        self._keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.keyValueHeads * config.headDim,
            bias: config.attentionBias
        )
        self._valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.keyValueHeads * config.headDim,
            bias: config.attentionBias
        )
        self._outputProjection.wrappedValue = Linear(
            config.attentionHeads * config.headDim,
            config.hiddenSize,
            bias: config.attentionBias
        )
    }

    func projectedKeyValues(
        _ input: MLXArray,
        offset: Int
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)

        let queries = rope(
            queryProjection(input)
                .reshaped(batchSize, sequenceLength, attentionHeads, headDim)
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        let keys = rope(
            keyProjection(input)
                .reshaped(batchSize, sequenceLength, keyValueHeads, headDim)
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        let values = valueProjection(input)
            .reshaped(batchSize, sequenceLength, keyValueHeads, headDim)
            .transposed(0, 2, 1, 3)
        return (queries, keys, values)
    }

    func attend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }

    func output(_ attended: MLXArray, batchSize: Int, sequenceLength: Int) -> MLXArray {
        outputProjection(
            attended
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, sequenceLength, -1)
        )
    }
}

private final class IQuestLoopCoderMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: IQuestLoopCoderConfiguration) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class IQuestLoopCoderBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: IQuestLoopCoderAttention
    @ModuleInfo(key: "mlp") private var mlp: IQuestLoopCoderMLP
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: IQuestLoopCoderConfiguration) {
        self._attention.wrappedValue = IQuestLoopCoderAttention(config)
        self._mlp.wrappedValue = IQuestLoopCoderMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func loopInput(_ input: MLXArray) -> MLXArray {
        inputLayerNorm(input)
    }

    func output(
        _ input: MLXArray,
        attended: MLXArray,
        batchSize: Int,
        sequenceLength: Int
    ) -> MLXArray {
        var hidden = input + attention.output(
            attended,
            batchSize: batchSize,
            sequenceLength: sequenceLength
        )
        hidden = hidden + mlp(postAttentionLayerNorm(hidden))
        return hidden
    }
}

private final class IQuestLoopCoderBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [IQuestLoopCoderBlock]
    @ModuleInfo(key: "norm") private var norm: RMSNorm
    @ModuleInfo(key: "gate_projections") fileprivate var gateProjections: [IQuestLoopGateProjection]

    private let loopWindowSize: Int

    init(_ config: IQuestLoopCoderConfiguration) {
        precondition(config.loopCount == 2, "iQuest Loop Coder supports loop_num=2")
        self.loopWindowSize = config.loopWindowSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            IQuestLoopCoderBlock(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._gateProjections.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            IQuestLoopGateProjection(headCount: config.attentionHeads, headDim: config.headDim)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batchSize = inputs.dim(0)
        let sequenceLength = inputs.dim(1)
        var hiddenStates = embedTokens(inputs)

        let layerCount = layers.count
        let fullCaches = cache.map { Array($0.prefix(layerCount)) }
        let localCaches = cache.map { Array($0.dropFirst(layerCount)) }
        let fullMask = createAttentionMask(h: hiddenStates, cache: fullCaches?.first)
        let localMask = createAttentionMask(
            h: hiddenStates,
            cache: localCaches?.first,
            windowSize: loopWindowSize
        )

        var firstLoopKeyValues: [(keys: MLXArray, values: MLXArray)] = []
        firstLoopKeyValues.reserveCapacity(layerCount)

        for (index, layer) in layers.enumerated() {
            let normalized = layer.loopInput(hiddenStates)
            let fullCache = fullCaches?[index]
            let projected = layer.attention.projectedKeyValues(
                normalized,
                offset: fullCache?.offset ?? 0
            )
            let keysAndValues: (keys: MLXArray, values: MLXArray)
            if let fullCache {
                keysAndValues = updateCacheReturningMaterializedKV(
                    keys: projected.keys,
                    values: projected.values,
                    cache: fullCache
                )
            } else {
                keysAndValues = (projected.keys, projected.values)
            }
            firstLoopKeyValues.append(keysAndValues)

            let attended = layer.attention.attend(
                queries: projected.queries,
                keys: keysAndValues.keys,
                values: keysAndValues.values,
                mask: fullMask
            )
            hiddenStates = layer.output(
                hiddenStates,
                attended: attended,
                batchSize: batchSize,
                sequenceLength: sequenceLength
            )
        }

        for index in layers.indices {
            let layer = layers[index]
            let normalized = layer.loopInput(hiddenStates)
            let localCache = localCaches?[index]
            let projected = layer.attention.projectedKeyValues(
                normalized,
                offset: localCache?.offset ?? 0
            )
            let gate = gateProjections[index](projected.queries)
            let globalAttention = layer.attention.attend(
                queries: projected.queries,
                keys: firstLoopKeyValues[index].keys,
                values: firstLoopKeyValues[index].values,
                mask: fullMask
            )

            let localKeyValues: (keys: MLXArray, values: MLXArray)
            if let localCache {
                localKeyValues = updateCacheReturningMaterializedKV(
                    keys: projected.keys,
                    values: projected.values,
                    cache: localCache
                )
            } else {
                localKeyValues = (projected.keys, projected.values)
            }
            let localAttention = layer.attention.attend(
                queries: projected.queries,
                keys: localKeyValues.keys,
                values: localKeyValues.values,
                mask: localMask
            )
            let mixed = gate * globalAttention + (1 - gate) * localAttention
            hiddenStates = layer.output(
                hiddenStates,
                attended: mixed,
                batchSize: batchSize,
                sequenceLength: sequenceLength
            )
        }

        return norm(hiddenStates)
    }
}

internal final class IQuestLoopCoderModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let configuration: IQuestLoopCoderConfiguration
    private let model: IQuestLoopCoderBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: IQuestLoopCoderConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(
            repeating: configuration.keyValueHeads,
            count: configuration.hiddenLayers * 2
        )
        self.model = IQuestLoopCoderBackbone(configuration)
        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
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
        let hiddenState = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenState), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let fullCaches = (0 ..< configuration.hiddenLayers).map { _ in
            KVCacheSimple() as KVCache
        }
        let localCaches = (0 ..< configuration.hiddenLayers).map { _ in
            RotatingKVCache(maxSize: configuration.loopWindowSize) as KVCache
        }
        return fullCaches + localCaches
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .block(),
            sidecarPolicy: .dropActivationScale
        ).weights
        sanitized = sanitized.filter { key, _ in
            !key.contains(".rotary_emb.inv_freq")
                && (!configuration.tieWordEmbeddings || key != "lm_head.weight")
        }
        return sanitized
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "k_proj", "v_proj", "o_proj"]) }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}
