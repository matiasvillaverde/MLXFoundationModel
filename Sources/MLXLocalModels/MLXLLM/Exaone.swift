import Foundation
import MLX
import MLXFast
import MLXNN

internal struct ExaoneConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var vocabularySize: Int
    internal var ropeTheta: Float
    internal var layerNormEpsilon: Float
    internal var kvHeads: Int
    internal var headDim: Int?
    internal var maxPositionEmbeddings: Int?
    internal var ropeTraditional: Bool
    internal var ropeScaling: [String: StringOrNumber]?
    internal var tieWordEmbeddings: Bool
    internal var attentionBias: Bool
    internal var mlpBias: Bool

    internal var resolvedHeadDim: Int {
        headDim ?? hiddenSize / attentionHeads
    }

    internal init(
        modelType: String = "exaone",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        vocabularySize: Int,
        ropeTheta: Float = 1_000_000,
        layerNormEpsilon: Float = 1e-5,
        kvHeads: Int,
        headDim: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false,
        mlpBias: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.layerNormEpsilon = layerNormEpsilon
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case layerNormEpsilon = "layer_norm_epsilon"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "exaone",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            ropeTheta: try container.decode(Float.self, forKey: .ropeTheta),
            layerNormEpsilon: try container.decode(Float.self, forKey: .layerNormEpsilon),
            kvHeads: try container.decode(Int.self, forKey: .kvHeads),
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                ?? false,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? true,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        )
    }
}

internal struct ExaoneAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headDim: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: ExaoneConfiguration) {
        precondition(config.hiddenSize > 0, "EXAONE hidden_size must be positive")
        precondition(config.attentionHeads > 0, "EXAONE num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "EXAONE num_key_value_heads must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "EXAONE attention heads must be divisible by KV heads"
        )
        precondition(config.resolvedHeadDim > 0, "EXAONE head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDim = config.resolvedHeadDim
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.attentionScale = 1 / Float(headDim).squareRoot()
    }
}

private final class ExaoneAttentionCore: Module {
    private let layout: ExaoneAttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "out_proj") private var outputProjection: Linear

    init(_ config: ExaoneConfiguration) {
        let layout = ExaoneAttentionLayout(config)
        self.layout = layout
        self.rope = initializeRope(
            dims: layout.headDim,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        _queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        _keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _outputProjection.wrappedValue = Linear(
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
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(hiddenStates)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, layout.queryProjectionSize)

        return outputProjection(output)
    }
}

private final class ExaoneAttention: Module {
    @ModuleInfo(key: "attention") fileprivate var core: ExaoneAttentionCore

    init(_ config: ExaoneConfiguration) {
        _core.wrappedValue = ExaoneAttentionCore(config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        core(hiddenStates, mask: mask, cache: cache)
    }
}

private final class ExaoneFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc_0") private var gateProjection: Linear
    @ModuleInfo(key: "c_fc_1") private var upProjection: Linear
    @ModuleInfo(key: "c_proj") private var downProjection: Linear

    init(_ config: ExaoneConfiguration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class ExaoneTransformerBlock: Module {
    @ModuleInfo(key: "ln_1") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "attn") fileprivate var attention: ExaoneAttention
    @ModuleInfo(key: "ln_2") private var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") private var feedForward: ExaoneFeedForward

    init(_ config: ExaoneConfiguration) {
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _attention.wrappedValue = ExaoneAttention(config)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _feedForward.wrappedValue = ExaoneFeedForward(config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = hiddenStates
            + attention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class ExaoneBackbone: Module {
    @ModuleInfo(key: "wte") var tokenEmbedding: Embedding
    @ModuleInfo(key: "h") fileprivate var layers: [ExaoneTransformerBlock]
    @ModuleInfo(key: "ln_f") private var finalLayerNorm: RMSNorm

    init(_ config: ExaoneConfiguration) {
        precondition(config.vocabularySize > 0, "EXAONE vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "EXAONE num_layers must be positive")

        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            ExaoneTransformerBlock(config)
        }
        _finalLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbedding(inputs)
        let cacheArray: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let mask = createAttentionMask(h: hiddenStates, cache: cacheArray[0])

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cacheArray[index])
        }

        return finalLayerNorm(hiddenStates)
    }
}

internal final class ExaoneModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let configuration: ExaoneConfiguration

    @ModuleInfo(key: "transformer") private var transformer: ExaoneBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: ExaoneConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0 ..< configuration.hiddenLayers).map { _ in configuration.kvHeads }
        _transformer.wrappedValue = ExaoneBackbone(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var output = transformer(inputs, cache: cache)
        if let lmHead {
            output = lmHead(output)
        } else {
            output = transformer.tokenEmbedding.asLinear(output)
        }
        return output
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        var logits = lastTokenHiddenState(transformer(input[text: .newAxis].tokens, cache: cache))
        if let lmHead {
            logits = lmHead(logits)
        } else {
            logits = transformer.tokenEmbedding.asLinear(logits)
        }
        return greedyTokenOutput(logits: logits, state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["lm_head.scales"] = nil
            weights["lm_head.biases"] = nil
        }
        return weights
    }
}

extension ExaoneModel: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        transformer.layers.map { ($0.attention.core, ["q_proj", "v_proj"]) }
    }
}
