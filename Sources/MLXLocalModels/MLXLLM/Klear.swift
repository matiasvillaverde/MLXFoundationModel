import Foundation
import MLX
import MLXNN

internal struct KlearConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var attentionBias: Bool
    var mlpOnlyLayers: [Int]
    var numExperts: Int
    var numExpertsPerToken: Int
    var decoderSparseStep: Int
    var numSharedExperts: Int
    var normTopkProb: Bool
    var ropeTheta: Float
    var maxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool
    var hiddenActivation: String

    internal init(
        modelType: String = "Klear",
        vocabularySize: Int,
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        rmsNormEps: Float = 1e-5,
        attentionBias: Bool = false,
        mlpOnlyLayers: [Int] = [],
        numExperts: Int,
        numExpertsPerToken: Int,
        decoderSparseStep: Int = 1,
        numSharedExperts: Int = 1,
        normTopkProb: Bool = true,
        ropeTheta: Float = 500_000,
        maxPositionEmbeddings: Int = 65_536,
        tieWordEmbeddings: Bool = false,
        hiddenActivation: String = "silu"
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.rmsNormEps = rmsNormEps
        self.attentionBias = attentionBias
        self.mlpOnlyLayers = mlpOnlyLayers
        self.numExperts = numExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.decoderSparseStep = decoderSparseStep
        self.numSharedExperts = numSharedExperts
        self.normTopkProb = normTopkProb
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
        self.hiddenActivation = hiddenActivation
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case mlpOnlyLayers = "mlp_only_layers"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case numSharedExperts = "n_shared_experts"
        case normTopkProb = "norm_topk_prob"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenActivation = "hidden_act"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "Klear",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            moeIntermediateSize: try container.decode(Int.self, forKey: .moeIntermediateSize),
            attentionHeads: attentionHeads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            mlpOnlyLayers: try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers)
                ?? [],
            numExperts: try container.decode(Int.self, forKey: .numExperts),
            numExpertsPerToken: try container.decode(Int.self, forKey: .numExpertsPerToken),
            decoderSparseStep: try container.decodeIfPresent(
                Int.self,
                forKey: .decoderSparseStep
            ) ?? 1,
            numSharedExperts: try container.decodeIfPresent(
                Int.self,
                forKey: .numSharedExperts
            ) ?? 1,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? true,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 500_000,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 65_536,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            hiddenActivation: try container.decodeIfPresent(String.self, forKey: .hiddenActivation)
                ?? "silu"
        )
    }
}

internal struct KlearAttentionLayout: Sendable, Equatable {
    let hiddenSize: Int
    let queryHeads: Int
    let keyValueHeads: Int
    let headSize: Int
    let queryDimensions: Int
    let keyValueDimensions: Int
    let scale: Float

    init(_ config: KlearConfiguration) {
        precondition(config.hiddenSize > 0, "Klear hidden_size must be positive")
        precondition(config.attentionHeads > 0, "Klear num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "Klear num_key_value_heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "Klear hidden_size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "Klear attention heads must group key/value heads"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = config.hiddenSize / config.attentionHeads
        self.queryDimensions = queryHeads * headSize
        self.keyValueDimensions = keyValueHeads * headSize
        self.scale = pow(Float(headSize), -0.5)
    }
}

internal struct KlearLayerPlan: Sendable, Equatable {
    let usesSparseExperts: [Bool]

    init(_ config: KlearConfiguration) {
        precondition(config.hiddenLayers > 0, "Klear num_hidden_layers must be positive")
        precondition(config.decoderSparseStep > 0, "Klear decoder_sparse_step must be positive")
        precondition(config.numExpertsPerToken > 0, "Klear num_experts_per_tok must be positive")
        precondition(config.numExperts > 0, "Klear num_experts must be positive")
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "Klear selected experts cannot exceed available experts"
        )

        self.usesSparseExperts = (0 ..< config.hiddenLayers).map { layerIndex in
            !config.mlpOnlyLayers.contains(layerIndex)
                && config.numExperts > 0
                && (layerIndex + 1).isMultiple(of: config.decoderSparseStep)
        }
    }

    func isSparse(layerIndex: Int) -> Bool {
        usesSparseExperts[layerIndex]
    }
}

internal struct KlearRoutingPlan: Sendable, Equatable {
    let numExperts: Int
    let expertsPerToken: Int
    let normalizeTopK: Bool

    init(_ config: KlearConfiguration) {
        precondition(config.numExperts > 0, "Klear num_experts must be positive")
        precondition(config.numExpertsPerToken > 0, "Klear num_experts_per_tok must be positive")
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "Klear selected experts cannot exceed available experts"
        )

        self.numExperts = config.numExperts
        self.expertsPerToken = config.numExpertsPerToken
        self.normalizeTopK = config.normTopkProb
    }

    func route(logits: MLXArray, expertBias: MLXArray, outputDType: DType) -> (
        indices: MLXArray, scores: MLXArray
    ) {
        let routingWeights = MLX.sigmoid(logits.asType(.float32))
        let biasedWeights = routingWeights + expertBias.reshaped(1, 1, numExperts)
        let indices = MLX.argPartition(
            -biasedWeights,
            kth: expertsPerToken - 1,
            axis: -1
        )[.ellipsis, ..<expertsPerToken]
        var scores = MLX.takeAlong(routingWeights, indices, axis: -1)
        if normalizeTopK {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }
        return (indices, scores.asType(outputDType))
    }
}

private final class KlearAttention: Module {
    private let layout: KlearAttentionLayout
    private let rope: RoPE

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") private var keyNorm: RMSNorm

    init(_ config: KlearConfiguration) {
        self.layout = KlearAttentionLayout(config)
        self.rope = RoPE(
            dimensions: layout.headSize,
            traditional: false,
            base: config.ropeTheta
        )

        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryDimensions,
            bias: config.attentionBias
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueDimensions,
            bias: config.attentionBias
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueDimensions,
            bias: config.attentionBias
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryDimensions,
            config.hiddenSize,
            bias: config.attentionBias
        )
        _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headSize, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: layout.headSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)

        var queries = queryProjection(hiddenStates)
        var keys = keyProjection(hiddenStates)
        var values = valueProjection(hiddenStates)

        queries = queryNorm(
            queries.reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
        )
        .transposed(0, 2, 1, 3)
        keys = keyNorm(
            keys.reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
        )
        .transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(attended)
    }
}

private final class KlearMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProjection.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProjection.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProjection.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class KlearSparseMoEBlock: Module, UnaryLayer {
    private let routing: KlearRoutingPlan

    @ModuleInfo(key: "gate") private var gate: Linear
    @ModuleInfo(key: "experts") private var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") private var sharedExperts: KlearMLP
    @ModuleInfo(key: "coefficient") private var coefficient: Linear
    @ParameterInfo(key: "expert_bias") private var expertBias: MLXArray

    init(_ config: KlearConfiguration) {
        self.routing = KlearRoutingPlan(config)

        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        _sharedExperts.wrappedValue = KlearMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.moeIntermediateSize * config.numSharedExperts
        )
        _coefficient.wrappedValue = Linear(config.hiddenSize, 2, bias: true)
        _expertBias.wrappedValue = MLXArray.zeros([config.numExperts], dtype: .float32)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let routed = routing.route(
            logits: gate(hiddenStates),
            expertBias: expertBias,
            outputDType: hiddenStates.dtype
        )
        let expertOutput = experts(hiddenStates, routed.indices)
        let combinedExperts = (expertOutput * routed.scores[.ellipsis, .newAxis])
            .sum(axis: -2)
        let coefficients = MLX.softmax(coefficient(hiddenStates), axis: -1, precise: true)
        return combinedExperts * coefficients[.ellipsis, ..<1]
            + sharedExperts(hiddenStates) * coefficients[.ellipsis, 1...]
    }
}

private final class KlearDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: KlearAttention
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ config: KlearConfiguration, layerIndex: Int, layerPlan: KlearLayerPlan) {
        _attention.wrappedValue = KlearAttention(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )

        if layerPlan.isSparse(layerIndex: layerIndex) {
            self.mlp = KlearSparseMoEBlock(config)
        } else {
            self.mlp = KlearMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize
            )
        }
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = attention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        let withAttention = hiddenStates + attended
        return withAttention + mlp(postAttentionLayerNorm(withAttention))
    }
}

private final class KlearModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [KlearDecoderLayer]
    @ModuleInfo(key: "norm") private var outputNorm: RMSNorm

    init(_ config: KlearConfiguration) {
        let layerPlan = KlearLayerPlan(config)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { layerIndex in
            KlearDecoderLayer(config, layerIndex: layerIndex, layerPlan: layerPlan)
        }
        _outputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return outputNorm(hiddenStates)
    }
}

internal final class KlearModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    private let configuration: KlearConfiguration
    private let model: KlearModelInner

    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: KlearConfiguration) {
        precondition(
            configuration.hiddenActivation == "silu",
            "Klear currently supports the published silu activation"
        )
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self.model = KlearModelInner(configuration)

        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hiddenStates = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        if let lmHead {
            return greedyTokenOutput(logits: lmHead(hiddenStates), state: state)
        }
        return greedyTokenOutput(logits: model.embedTokens.asLinear(hiddenStates), state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }

        return sanitized
    }
}

extension KlearModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
