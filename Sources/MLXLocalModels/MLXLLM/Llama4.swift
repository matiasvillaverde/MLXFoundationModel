import MLX
import MLXFast
import MLXNN

internal struct Llama4TextConfiguration: Codable, Equatable, Sendable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var attentionHeads: Int
    internal var hiddenLayers: Int
    internal var vocabularySize: Int
    internal var intermediateSize: Int
    internal var denseIntermediateSize: Int
    internal var keyValueHeads: Int
    internal var rmsNormEps: Float
    internal var ropeTheta: Float
    internal var ropeScaling: [String: StringOrNumber]?
    internal var headDimensions: Int
    internal var tieWordEmbeddings: Bool
    internal var attentionBias: Bool
    internal var noRopeLayers: [Bool]
    internal var useQKNorm: Bool
    internal var attentionChunkSize: Int
    internal var interleaveMoELayerStep: Int
    internal var expertsPerToken: Int
    internal var localExperts: Int
    internal var moeLayers: [Int]
    internal var attentionTemperatureTuning: Bool
    internal var floorScale: Int
    internal var attentionScale: Float
    internal var maxPositionEmbeddings: Int?

    internal init(
        modelType: String = "llama4_text",
        hiddenSize: Int,
        attentionHeads: Int,
        hiddenLayers: Int,
        vocabularySize: Int,
        intermediateSize: Int,
        denseIntermediateSize: Int? = nil,
        keyValueHeads: Int,
        rmsNormEps: Float,
        ropeTheta: Float = 500_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        headDimensions: Int? = nil,
        tieWordEmbeddings: Bool = false,
        attentionBias: Bool = false,
        noRopeLayers: [Bool]? = nil,
        useQKNorm: Bool = false,
        attentionChunkSize: Int = 8_192,
        interleaveMoELayerStep: Int = 1,
        expertsPerToken: Int = 1,
        localExperts: Int = 1,
        moeLayers: [Int] = [],
        attentionTemperatureTuning: Bool = true,
        floorScale: Int = 8_192,
        attentionScale: Float = 0.1,
        maxPositionEmbeddings: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.attentionHeads = attentionHeads
        self.hiddenLayers = hiddenLayers
        self.vocabularySize = vocabularySize
        self.intermediateSize = intermediateSize
        self.denseIntermediateSize = denseIntermediateSize ?? intermediateSize
        self.keyValueHeads = keyValueHeads
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.headDimensions = headDimensions ?? hiddenSize / attentionHeads
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.noRopeLayers = noRopeLayers ?? Array(repeating: true, count: hiddenLayers)
        self.useQKNorm = useQKNorm
        self.attentionChunkSize = attentionChunkSize
        self.interleaveMoELayerStep = interleaveMoELayerStep
        self.expertsPerToken = expertsPerToken
        self.localExperts = localExperts
        self.moeLayers = moeLayers
        self.attentionTemperatureTuning = attentionTemperatureTuning
        self.floorScale = floorScale
        self.attentionScale = attentionScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    internal var resolvedNoRopeLayers: [Bool] {
        if noRopeLayers.count == hiddenLayers {
            return noRopeLayers
        }
        return Array(repeating: true, count: hiddenLayers)
    }

    internal func usesRope(at layerIndex: Int) -> Bool {
        resolvedNoRopeLayers[layerIndex]
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "intermediate_size_mlp"
        case keyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case headDimensions = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case noRopeLayers = "no_rope_layers"
        case useQKNorm = "use_qk_norm"
        case attentionChunkSize = "attention_chunk_size"
        case interleaveMoELayerStep = "interleave_moe_layer_step"
        case expertsPerToken = "num_experts_per_tok"
        case localExperts = "num_local_experts"
        case moeLayers = "moe_layers"
        case attentionTemperatureTuning = "attn_temperature_tuning"
        case floorScale = "floor_scale"
        case attentionScale = "attn_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        let ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeParameters
        )
        let ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeScaling
        ) ?? ropeParameters

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "llama4_text",
            hiddenSize: hiddenSize,
            attentionHeads: attentionHeads,
            hiddenLayers: hiddenLayers,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            intermediateSize: intermediateSize,
            denseIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .denseIntermediateSize
            ) ?? intermediateSize,
            keyValueHeads: try container.decode(Int.self, forKey: .keyValueHeads),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? ropeParameters?["rope_theta"]?.asFloat()
                ?? 500_000,
            ropeScaling: ropeScaling,
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions)
                ?? hiddenSize / attentionHeads,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            noRopeLayers: try Self.decodeNoRopeLayers(from: container, count: hiddenLayers),
            useQKNorm: try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false,
            attentionChunkSize: try container.decodeIfPresent(
                Int.self,
                forKey: .attentionChunkSize
            ) ?? 8_192,
            interleaveMoELayerStep: try container.decodeIfPresent(
                Int.self,
                forKey: .interleaveMoELayerStep
            ) ?? 1,
            expertsPerToken: try container.decodeIfPresent(Int.self, forKey: .expertsPerToken)
                ?? 1,
            localExperts: try container.decodeIfPresent(Int.self, forKey: .localExperts) ?? 1,
            moeLayers: try container.decodeIfPresent([Int].self, forKey: .moeLayers) ?? [],
            attentionTemperatureTuning: try container.decodeIfPresent(
                Bool.self,
                forKey: .attentionTemperatureTuning
            ) ?? true,
            floorScale: try container.decodeIfPresent(Int.self, forKey: .floorScale) ?? 8_192,
            attentionScale: try container.decodeIfPresent(Float.self, forKey: .attentionScale)
                ?? 0.1,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            )
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(denseIntermediateSize, forKey: .denseIntermediateSize)
        try container.encode(keyValueHeads, forKey: .keyValueHeads)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encode(headDimensions, forKey: .headDimensions)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(noRopeLayers, forKey: .noRopeLayers)
        try container.encode(useQKNorm, forKey: .useQKNorm)
        try container.encode(attentionChunkSize, forKey: .attentionChunkSize)
        try container.encode(interleaveMoELayerStep, forKey: .interleaveMoELayerStep)
        try container.encode(expertsPerToken, forKey: .expertsPerToken)
        try container.encode(localExperts, forKey: .localExperts)
        try container.encode(moeLayers, forKey: .moeLayers)
        try container.encode(attentionTemperatureTuning, forKey: .attentionTemperatureTuning)
        try container.encode(floorScale, forKey: .floorScale)
        try container.encode(attentionScale, forKey: .attentionScale)
        try container.encodeIfPresent(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
    }

    private static func decodeNoRopeLayers(
        from container: KeyedDecodingContainer<CodingKeys>,
        count: Int
    ) throws -> [Bool]? {
        guard let raw = try container.decodeIfPresent(
            [StringOrNumber].self,
            forKey: .noRopeLayers
        ) else {
            return nil
        }
        if raw.count != count {
            return nil
        }
        return raw.map { value in
            if let bool = value.asBool() {
                return bool
            }
            return (value.asInt() ?? 0) != 0
        }
    }
}

internal struct Llama4Configuration: Codable, Equatable, Sendable {
    internal var modelType: String
    internal var textConfig: Llama4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    internal init(modelType: String = "llama4", textConfig: Llama4TextConfiguration) {
        self.modelType = modelType
        self.textConfig = textConfig
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "llama4"
        self.textConfig = try container.decode(Llama4TextConfiguration.self, forKey: .textConfig)
    }
}

internal struct Llama4AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: Llama4TextConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.keyValueHeads > 0, "num_key_value_heads must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.keyValueHeads),
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        precondition(config.headDimensions > 0, "head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.headSize = config.headDimensions
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

private enum Llama4FeedForwardKind {
    case dense
    case moe
}

private final class Llama4SelfAttention: Module {
    private let layout: Llama4AttentionLayout
    private let rope: RoPELayer?
    private let useQKNorm: Bool
    private let qkNormEps: Float

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: Llama4TextConfiguration, useRope: Bool) {
        let layout = Llama4AttentionLayout(config)
        self.layout = layout
        self.useQKNorm = config.useQKNorm
        self.qkNormEps = config.rmsNormEps
        self.rope = useRope
            ? initializeRope(
                dims: layout.headSize,
                base: config.ropeTheta,
                traditional: true,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings
            )
            : nil

        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: config.attentionBias
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        var queries = queryProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)

        if useQKNorm {
            queries = MLXFast.rmsNorm(queries, weight: MLXArray.mlxNone, eps: qkNormEps)
            keys = MLXFast.rmsNorm(keys, weight: MLXArray.mlxNone, eps: qkNormEps)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        let values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        if let rope {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        let attentionOutput = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(attentionOutput)
    }
}

private final class Llama4DenseFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: Llama4TextConfiguration, intermediateSize: Int) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            intermediateSize,
            bias: false
        )
        self._downProjection.wrappedValue = Linear(
            intermediateSize,
            config.hiddenSize,
            bias: false
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            intermediateSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class Llama4MoEFeedForward: Module, UnaryLayer {
    private let expertsPerToken: Int

    @ModuleInfo(key: "experts") private var experts: SwitchGLU
    @ModuleInfo(key: "router") private var router: Linear
    @ModuleInfo(key: "shared_expert") private var sharedExpert: Llama4DenseFeedForward

    init(_ config: Llama4TextConfiguration) {
        precondition(config.expertsPerToken == 1, "Llama 4 MoE currently expects top-1 routing")
        self.expertsPerToken = config.expertsPerToken
        self._experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.localExperts
        )
        self._router.wrappedValue = Linear(
            config.hiddenSize,
            config.localExperts,
            bias: false
        )
        self._sharedExpert.wrappedValue = Llama4DenseFeedForward(
            config,
            intermediateSize: config.intermediateSize
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let logits = router(hiddenStates)
        let indices = argPartition(
            -logits,
            kth: expertsPerToken - 1,
            axis: -1
        )[.ellipsis, ..<expertsPerToken]
        let scores = sigmoid(takeAlong(logits, indices, axis: -1))
        return experts(hiddenStates * scores, indices).sum(axis: -2) + sharedExpert(hiddenStates)
    }
}

private final class Llama4TransformerBlock: Module {
    internal let useChunkedAttention: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Llama4SelfAttention
    @ModuleInfo(key: "feed_forward") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(
        _ config: Llama4TextConfiguration,
        layerIndex: Int,
        feedForwardKind: Llama4FeedForwardKind,
        useChunkedAttention: Bool
    ) {
        self.useChunkedAttention = useChunkedAttention
        self._selfAttention.wrappedValue = Llama4SelfAttention(
            config,
            useRope: config.usesRope(at: layerIndex)
        )
        switch feedForwardKind {
        case .dense:
            self._feedForward.wrappedValue = Llama4DenseFeedForward(
                config,
                intermediateSize: config.denseIntermediateSize
            )
        case .moe:
            self._feedForward.wrappedValue = Llama4MoEFeedForward(config)
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = hiddenStates
            + selfAttention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class Llama4Backbone: Module {
    private let config: Llama4TextConfiguration
    private let usesChunkedMasks: Bool

    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [Llama4TransformerBlock]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    init(_ config: Llama4TextConfiguration, wrapperSemantics: Bool) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "num_hidden_layers must be positive")

        self.config = config
        self.usesChunkedMasks = wrapperSemantics
        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { index in
            let isInterleavedMoELayer = wrapperSemantics
                && config.interleaveMoELayerStep > 0
                && (index % config.interleaveMoELayerStep == config.interleaveMoELayerStep - 1)
            let isMoELayer = config.moeLayers.contains(index) || isInterleavedMoELayer
            return Llama4TransformerBlock(
                config,
                layerIndex: index,
                feedForwardKind: isMoELayer ? .moe : .dense,
                useChunkedAttention: wrapperSemantics && config.usesRope(at: index)
            )
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        let fullMask = createAttentionMask(h: hiddenStates, cache: cache)
        let chunkedMask = usesChunkedMasks
            ? createAttentionMask(
                h: hiddenStates,
                cache: cache?.first,
                windowSize: config.attentionChunkSize
            )
            : fullMask

        for (layerIndex, layer) in layers.enumerated() {
            let mask = layer.useChunkedAttention ? chunkedMask : fullMask
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

private struct Llama4WeightSanitizer {
    let prefix: String?
    let tieWordEmbeddings: Bool

    func callAsFunction(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter {
            !$0.key.contains("vision_model")
                && !$0.key.contains("vision_tower")
                && !$0.key.contains("multi_modal_projector")
                && !$0.key.contains("mm_projector")
                && !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
        sanitized = packExperts(in: sanitized)
        sanitized = normalizeOutputHead(in: sanitized)
        if tieWordEmbeddings {
            sanitized[prefix.map { "\($0).output.weight" } ?? "output.weight"] = nil
            sanitized[prefix.map { "\($0).lm_head.weight" } ?? "lm_head.weight"] = nil
        }
        return sanitized
    }

    private func normalizeOutputHead(in weights: [String: MLXArray]) -> [String: MLXArray] {
        var normalized = weights
        let outputKey = prefix.map { "\($0).output.weight" } ?? "output.weight"
        let headKey = prefix.map { "\($0).lm_head.weight" } ?? "lm_head.weight"
        if normalized[headKey] == nil, let output = normalized[outputKey] {
            normalized[headKey] = output
            normalized[outputKey] = nil
        }
        return normalized
    }

    private func packExperts(in weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights
        for (key, tensor) in weights {
            guard key.hasSuffix(".experts.gate_up_proj") else {
                continue
            }
            let base = key.replacingOccurrences(of: ".gate_up_proj", with: "")
            let projections = split(tensor, parts: 2, axis: -1)
            packed[key] = nil
            packed["\(base).gate_proj.weight"] = projections[0].transposed(0, 2, 1)
            packed["\(base).up_proj.weight"] = projections[1].transposed(0, 2, 1)
        }
        for (key, tensor) in packed where key.hasSuffix(".experts.down_proj") {
            let destination = key.replacingOccurrences(
                of: ".down_proj",
                with: ".down_proj.weight"
            )
            packed[key] = nil
            packed[destination] = tensor.transposed(0, 2, 1)
        }
        return packed
    }
}

internal final class Llama4TextModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let config: Llama4TextConfiguration
    fileprivate let model: Llama4Backbone

    @ModuleInfo(key: "lm_head") private var output: Linear?

    internal init(_ config: Llama4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.keyValueHeads, count: config.hiddenLayers)
        self.model = Llama4Backbone(config, wrapperSemantics: false)

        if !config.tieWordEmbeddings {
            self._output.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
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
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Llama4WeightSanitizer(prefix: nil, tieWordEmbeddings: config.tieWordEmbeddings)(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let output {
            return output(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

internal final class Llama4Model: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let config: Llama4Configuration
    @ModuleInfo(key: "language_model") private var languageModel: Llama4LanguageModel

    internal init(_ config: Llama4Configuration) {
        self.config = config
        self.vocabularySize = config.textConfig.vocabularySize
        self.kvHeads = Array(
            repeating: config.textConfig.keyValueHeads,
            count: config.textConfig.hiddenLayers
        )
        self._languageModel.wrappedValue = Llama4LanguageModel(config.textConfig)
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

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Llama4WeightSanitizer(
            prefix: "language_model",
            tieWordEmbeddings: config.textConfig.tieWordEmbeddings
        )(weights)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

private final class Llama4LanguageModel: Module {
    private let config: Llama4TextConfiguration

    @ModuleInfo(key: "model") fileprivate var model: Llama4Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: Llama4TextConfiguration) {
        self.config = config
        self._model.wrappedValue = Llama4Backbone(config, wrapperSemantics: true)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false
            )
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { layer in
            layer.useChunkedAttention
                ? ChunkedKVCache(chunkSize: config.attentionChunkSize)
                : KVCacheSimple()
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

extension Llama4TextModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "k_proj", "v_proj", "o_proj"]) }
    }
}

extension Llama4Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        languageModel.model.layers.map {
            ($0.selfAttention, ["q_proj", "k_proj", "v_proj", "o_proj"])
        }
    }
}
