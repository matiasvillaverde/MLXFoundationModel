import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

internal struct GPTBigCodeConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int?
    var attentionHeads: Int
    var maxPositionEmbeddings: Int
    var layerNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int?
    var multiQuery: Bool
    var attentionBias: Bool
    var mlpBias: Bool
    var tieWordEmbeddings: Bool
    var activationFunction: String

    internal var feedForwardSize: Int {
        intermediateSize ?? 4 * hiddenSize
    }

    internal var resolvedKVHeads: Int {
        kvHeads ?? (multiQuery ? 1 : attentionHeads)
    }

    internal init(
        modelType: String = "gpt_bigcode",
        hiddenSize: Int = 768,
        hiddenLayers: Int = 20,
        intermediateSize: Int? = nil,
        attentionHeads: Int = 12,
        maxPositionEmbeddings: Int = 8_192,
        layerNormEps: Float = 1e-5,
        vocabularySize: Int = 49_152,
        kvHeads: Int? = nil,
        multiQuery: Bool = true,
        attentionBias: Bool = true,
        mlpBias: Bool = true,
        tieWordEmbeddings: Bool = true,
        activationFunction: String = "gelu_pytorch_tanh"
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.layerNormEps = layerNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.multiQuery = multiQuery
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.activationFunction = activationFunction
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "n_embd"
        case hiddenLayers = "n_layer"
        case intermediateSize = "n_inner"
        case attentionHeads = "n_head"
        case maxPositionEmbeddings = "n_positions"
        case layerNormEps = "layer_norm_epsilon"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case multiQuery = "multi_query"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case activationFunction = "activation_function"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? 12

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "gpt_bigcode",
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 20,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 8_192,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
                ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 49_152,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            multiQuery: try container.decodeIfPresent(Bool.self, forKey: .multiQuery)
                ?? true,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? true,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias)
                ?? true,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            activationFunction: try container.decodeIfPresent(
                String.self,
                forKey: .activationFunction
            ) ?? "gelu_pytorch_tanh"
        )
    }
}

// MARK: - Plans

internal struct GPTBigCodeAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let keyValueDimensions: Int
    internal let combinedProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: GPTBigCodeConfiguration) {
        precondition(config.hiddenSize > 0, "GPT-BigCode hidden size must be positive")
        precondition(config.attentionHeads > 0, "GPT-BigCode attention heads must be positive")
        precondition(config.resolvedKVHeads > 0, "GPT-BigCode KV heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "GPT-BigCode hidden size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.resolvedKVHeads),
            "GPT-BigCode attention heads must group KV heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.resolvedKVHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.keyValueDimensions = keyValueHeads * headDimensions
        self.combinedProjectionSize = hiddenSize + 2 * keyValueDimensions
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }
}

internal struct GPTBigCodeWeightSanitizerPlan: Equatable, Sendable {
    internal let tieWordEmbeddings: Bool

    internal init(_ config: GPTBigCodeConfiguration) {
        self.tieWordEmbeddings = config.tieWordEmbeddings
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (checkpointKey, checkpointValue) in weights {
            let key = Self.normalizedKey(checkpointKey)
            guard !Self.shouldDrop(key) else {
                continue
            }
            if tieWordEmbeddings, Self.isOutputHead(key) {
                continue
            }
            sanitized[key] = checkpointValue
        }

        return sanitized
    }

    private static func normalizedKey(_ key: String) -> String {
        if key.hasPrefix("base_model.model.") {
            return String(key.dropFirst("base_model.model.".count))
        }
        if key.hasPrefix("model.transformer.") {
            return String(key.dropFirst("model.".count))
        }
        if key.hasPrefix("model.lm_head.") {
            return String(key.dropFirst("model.".count))
        }
        return key
    }

    private static func shouldDrop(_ key: String) -> Bool {
        key.hasSuffix(".attn.bias") || key.hasSuffix(".attn.masked_bias")
    }

    private static func isOutputHead(_ key: String) -> Bool {
        key == "lm_head.weight"
            || key == "lm_head.scales"
            || key == "lm_head.biases"
    }
}

// MARK: - Layers

internal final class GPTBigCodeAttention: Module {
    let layout: GPTBigCodeAttentionLayout

    @ModuleInfo(key: "c_attn") var combinedProjection: Linear
    @ModuleInfo(key: "c_proj") var outputProjection: Linear

    init(_ config: GPTBigCodeConfiguration) {
        self.layout = GPTBigCodeAttentionLayout(config)
        _combinedProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.combinedProjectionSize,
            bias: config.attentionBias
        )
        _outputProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
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
        let projected = combinedProjection(hiddenStates)
        let qkv = split(
            projected,
            indices: [layout.hiddenSize, layout.hiddenSize + layout.keyValueDimensions],
            axis: -1
        )

        let queries = qkv[0]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let keys = qkv[1]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = qkv[2]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

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

internal final class GPTBigCodeFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var upProjection: Linear
    @ModuleInfo(key: "c_proj") var downProjection: Linear

    let activation: String

    init(_ config: GPTBigCodeConfiguration) {
        self.activation = config.activationFunction.lowercased()
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.feedForwardSize,
            bias: config.mlpBias
        )
        _downProjection.wrappedValue = Linear(
            config.feedForwardSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let projected = upProjection(hiddenStates)
        switch activation {
        case "gelu_fast", "gelu_new", "gelu_pytorch_tanh":
            return downProjection(geluApproximate(projected))
        default:
            return downProjection(gelu(projected))
        }
    }
}

internal final class GPTBigCodeTransformerBlock: Module {
    @ModuleInfo(key: "attn") var attention: GPTBigCodeAttention
    @ModuleInfo(key: "mlp") var feedForward: GPTBigCodeFeedForward
    @ModuleInfo(key: "ln_1") var attentionLayerNorm: LayerNorm
    @ModuleInfo(key: "ln_2") var feedForwardLayerNorm: LayerNorm

    init(_ config: GPTBigCodeConfiguration) {
        _attention.wrappedValue = GPTBigCodeAttention(config)
        _feedForward.wrappedValue = GPTBigCodeFeedForward(config)
        _attentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        _feedForwardLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttention = hiddenStates + attention(
            attentionLayerNorm(hiddenStates),
            mask: mask,
            cache: cache
        )
        return afterAttention + feedForward(feedForwardLayerNorm(afterAttention))
    }
}

internal final class GPTBigCodeBackbone: Module {
    @ModuleInfo(key: "wte") var tokenEmbedding: Embedding
    @ModuleInfo(key: "wpe") var positionEmbedding: Embedding
    @ModuleInfo(key: "h") var layers: [GPTBigCodeTransformerBlock]
    @ModuleInfo(key: "ln_f") var finalLayerNorm: LayerNorm

    init(_ config: GPTBigCodeConfiguration) {
        precondition(config.vocabularySize > 0, "GPT-BigCode vocabulary size must be positive")
        precondition(
            config.maxPositionEmbeddings > 0,
            "GPT-BigCode position embedding count must be positive"
        )
        precondition(config.hiddenLayers > 0, "GPT-BigCode must have at least one layer")

        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            GPTBigCodeTransformerBlock(config)
        }
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenCount = inputs.dim(1)
        let cacheOffset = cache?.first?.offset ?? 0
        let positionIDs = (MLXArray(0 ..< tokenCount) + MLXArray(cacheOffset))[.newAxis, 0...]

        var hiddenStates = tokenEmbedding(inputs) + positionEmbedding(positionIDs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }

        return finalLayerNorm(hiddenStates)
    }
}

internal final class GPTBigCodeModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: GPTBigCodeConfiguration

    @ModuleInfo(key: "transformer") var transformer: GPTBigCodeBackbone
    @ModuleInfo(key: "lm_head") var outputHead: Linear?

    init(_ config: GPTBigCodeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.resolvedKVHeads, count: config.hiddenLayers)
        _transformer.wrappedValue = GPTBigCodeBackbone(config)

        if !config.tieWordEmbeddings {
            _outputHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: transformer(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(
            transformer(input[text: .newAxis].tokens, cache: cache)
        )
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        GPTBigCodeWeightSanitizerPlan(configuration).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let outputHead {
            return outputHead(hiddenStates)
        }
        return transformer.tokenEmbedding.asLinear(hiddenStates)
    }
}

extension GPTBigCodeModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        transformer.layers.map { ($0.attention, ["c_attn", "c_proj"]) }
    }
}
