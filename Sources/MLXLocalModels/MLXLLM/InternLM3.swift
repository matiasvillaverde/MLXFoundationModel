import Darwin
import Foundation
import MLX
import MLXFast
import MLXNN

internal struct InternLM3Configuration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var intermediateSize: Int
    internal var attentionHeads: Int
    internal var rmsNormEps: Float
    internal var vocabularySize: Int
    internal var bias: Bool
    internal var qkvBias: Bool
    internal var maxPositionEmbeddings: Int
    internal var keyValueHeads: Int
    internal var ropeTheta: Float
    internal var ropeTraditional: Bool
    internal var ropeScaling: [String: StringOrNumber]?
    internal var tieWordEmbeddings: Bool
    internal var headDim: Int?

    internal var resolvedHeadDim: Int {
        headDim ?? hiddenSize / attentionHeads
    }

    internal init(
        modelType: String = "internlm3",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        bias: Bool = false,
        qkvBias: Bool = false,
        maxPositionEmbeddings: Int = 32_768,
        keyValueHeads: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        headDim: Int? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.bias = bias
        self.qkvBias = qkvBias
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.keyValueHeads = keyValueHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.headDim = headDim
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case bias
        case qkvBias = "qkv_bias"
        case maxPositionEmbeddings = "max_position_embeddings"
        case keyValueHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case headDim = "head_dim"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "internlm3",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            bias: try container.decodeIfPresent(Bool.self, forKey: .bias) ?? false,
            qkvBias: try container.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 32_768,
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads)
                ?? attentionHeads,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                ?? false,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? false,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim)
        )
    }
}

internal struct InternLM3RoPEPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let maxPositionEmbeddings: Int
    internal let traditional: Bool
    internal let base: Float
    internal let positionScale: Float
    internal let dynamicBaseScale: Float?

    internal init(_ config: InternLM3Configuration, dimensions: Int) {
        precondition(dimensions > 0, "InternLM3 rotary dimensions must be positive")
        precondition(config.maxPositionEmbeddings > 0, "max_position_embeddings must be positive")

        self.dimensions = dimensions
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.traditional = config.ropeTraditional
        self.base = config.ropeTheta

        let factor = config.ropeScaling?["factor"]?.asFloat() ?? 1
        precondition(factor > 0, "rope_scaling.factor must be positive")

        switch Self.ropeType(config.ropeScaling) {
        case "linear":
            self.positionScale = 1 / factor
            self.dynamicBaseScale = nil
        case "dynamic":
            precondition(dimensions > 2, "dynamic RoPE scaling requires dimensions greater than 2")
            self.positionScale = 2
            self.dynamicBaseScale = 2
        default:
            self.positionScale = 2
            self.dynamicBaseScale = nil
        }
    }

    internal func adjustedBase(sequenceLength: Int) -> Float {
        guard let dynamicBaseScale, sequenceLength > maxPositionEmbeddings else {
            return base
        }

        let scaledLength = dynamicBaseScale * Float(sequenceLength) / Float(maxPositionEmbeddings)
        let ratio = scaledLength - (dynamicBaseScale - 1)
        let exponent = Float(dimensions) / Float(dimensions - 2)
        return base * Darwin.powf(ratio, exponent)
    }

    private static func ropeType(_ scaling: [String: StringOrNumber]?) -> String? {
        guard let value = scaling?["type"] ?? scaling?["rope_type"],
              case .string(let string) = value else {
            return nil
        }
        return string
    }
}

private final class InternLM3RotaryEmbedding: Module {
    private let plan: InternLM3RoPEPlan

    init(_ plan: InternLM3RoPEPlan) {
        self.plan = plan
    }

    func callAsFunction(_ input: MLXArray, offset: Int = 0) -> MLXArray {
        let sequenceLength = input.dim(-2) + offset
        return MLXFast.RoPE(
            input,
            dimensions: plan.dimensions,
            traditional: plan.traditional,
            base: plan.adjustedBase(sequenceLength: sequenceLength),
            scale: plan.positionScale,
            offset: offset
        )
    }
}

internal struct InternLM3AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let keyValueGroups: Int
    internal let headDim: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: InternLM3Configuration) {
        precondition(config.hiddenSize > 0, "InternLM3 hidden_size must be positive")
        precondition(config.attentionHeads > 0, "InternLM3 num_attention_heads must be positive")
        precondition(config.keyValueHeads > 0, "InternLM3 num_key_value_heads must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.keyValueHeads),
            "InternLM3 attention heads must be divisible by KV heads"
        )
        precondition(config.resolvedHeadDim > 0, "InternLM3 head_dim must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.keyValueGroups = config.attentionHeads / config.keyValueHeads
        self.headDim = config.resolvedHeadDim
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.attentionScale = 1 / Float(headDim).squareRoot()
    }
}

private final class InternLM3Attention: Module {
    private let layout: InternLM3AttentionLayout
    private let rope: InternLM3RotaryEmbedding

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: InternLM3Configuration) {
        let layout = InternLM3AttentionLayout(config)
        self.layout = layout
        self.rope = InternLM3RotaryEmbedding(
            InternLM3RoPEPlan(config, dimensions: layout.headDim)
        )
        _queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.qkvBias
        )
        _keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.qkvBias
        )
        _valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.qkvBias
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: config.qkvBias
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

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

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

private final class InternLM3FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: InternLM3Configuration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.bias
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.bias
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.bias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class InternLM3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: InternLM3Attention
    @ModuleInfo(key: "mlp") private var feedForward: InternLM3FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: InternLM3Configuration) {
        _attention.wrappedValue = InternLM3Attention(config)
        _feedForward.wrappedValue = InternLM3FeedForward(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
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
            + attention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class InternLM3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [InternLM3TransformerBlock]
    @ModuleInfo private var norm: RMSNorm

    init(_ config: InternLM3Configuration) {
        precondition(config.vocabularySize > 0, "InternLM3 vocab_size must be positive")
        precondition(config.hiddenLayers > 0, "InternLM3 must have at least one layer")

        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            InternLM3TransformerBlock(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbeddings(inputs)
        let cacheArray: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let mask = createAttentionMask(h: hiddenStates, cache: cacheArray[0])

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cacheArray[index])
        }

        return norm(hiddenStates)
    }
}

internal final class InternLM3Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]
    internal let configuration: InternLM3Configuration

    @ModuleInfo(key: "model") private var model: InternLM3Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: InternLM3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(
            repeating: configuration.keyValueHeads,
            count: configuration.hiddenLayers
        )
        _model.wrappedValue = InternLM3Backbone(configuration)
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

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights.filter { key, _ in
            !key.contains("attention.rope.inv_freq")
                && !key.contains("self_attn.rope.inv_freq")
                && !key.contains("self_attn.rotary_emb.inv_freq")
        }
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

extension InternLM3Model: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
