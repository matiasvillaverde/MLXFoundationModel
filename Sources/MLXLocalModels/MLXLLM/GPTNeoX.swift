import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

internal struct GPTNeoXConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var maxPositionEmbeddings: Int
    var hiddenSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var layerNormEps: Float
    var vocabularySize: Int
    var rotaryEmbeddingBase: Float
    var rotaryPercent: Float
    var useParallelResidual: Bool
    var kvHeads: Int
    var intermediateSize: Int?
    var activationFunction: String
    var tieWordEmbeddings: Bool

    internal var feedForwardSize: Int {
        intermediateSize ?? 4 * hiddenSize
    }

    internal init(
        modelType: String = "gpt_neox",
        maxPositionEmbeddings: Int = 2_048,
        hiddenSize: Int = 512,
        attentionHeads: Int = 8,
        hiddenLayers: Int = 6,
        layerNormEps: Float = 1e-5,
        vocabularySize: Int = 50_304,
        rotaryEmbeddingBase: Float = 10_000,
        rotaryPercent: Float = 0.25,
        useParallelResidual: Bool = true,
        kvHeads: Int? = nil,
        intermediateSize: Int? = nil,
        activationFunction: String = "gelu",
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.hiddenSize = hiddenSize
        self.attentionHeads = attentionHeads
        self.hiddenLayers = hiddenLayers
        self.layerNormEps = layerNormEps
        self.vocabularySize = vocabularySize
        self.rotaryEmbeddingBase = rotaryEmbeddingBase
        self.rotaryPercent = rotaryPercent
        self.useParallelResidual = useParallelResidual
        self.kvHeads = kvHeads ?? attentionHeads
        self.intermediateSize = intermediateSize
        self.activationFunction = activationFunction
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case layerNormEps = "layer_norm_eps"
        case vocabularySize = "vocab_size"
        case rotaryEmbeddingBase = "rotary_emb_base"
        case rotaryPercent = "rotary_pct"
        case useParallelResidual = "use_parallel_residual"
        case kvHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case activationFunction = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? 8

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "gpt_neox",
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 2_048,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512,
            attentionHeads: attentionHeads,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 6,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
                ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 50_304,
            rotaryEmbeddingBase: try container.decodeIfPresent(
                Float.self,
                forKey: .rotaryEmbeddingBase
            ) ?? 10_000,
            rotaryPercent: try container.decodeIfPresent(Float.self, forKey: .rotaryPercent)
                ?? 0.25,
            useParallelResidual: try container.decodeIfPresent(
                Bool.self,
                forKey: .useParallelResidual
            ) ?? true,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize),
            activationFunction: try container.decodeIfPresent(
                String.self,
                forKey: .activationFunction
            ) ?? "gelu",
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

// MARK: - Plans

internal struct GPTNeoXAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let rotaryDimensions: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let combinedProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: GPTNeoXConfiguration) {
        precondition(config.hiddenSize > 0, "GPT-NeoX hidden size must be positive")
        precondition(config.attentionHeads > 0, "GPT-NeoX attention heads must be positive")
        precondition(config.kvHeads > 0, "GPT-NeoX KV heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "GPT-NeoX hidden size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "GPT-NeoX attention heads must group KV heads"
        )
        precondition(config.rotaryPercent > 0, "GPT-NeoX rotary percentage must be positive")

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.rotaryDimensions = max(1, Int(Float(headDimensions) * config.rotaryPercent))
        self.queryProjectionSize = queryHeads * headDimensions
        self.keyValueProjectionSize = keyValueHeads * headDimensions
        self.combinedProjectionSize = queryProjectionSize + 2 * keyValueProjectionSize
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }
}

private enum GPTNeoXRawIgnoredSuffix: String, CaseIterable {
    case attentionBias = ".attention.bias"
    case attentionMaskedBias = ".attention.masked_bias"
    case rotaryInverseFrequency = ".attention.rotary_emb.inv_freq"
}

internal struct GPTNeoXWeightSanitizerPlan: Equatable, Sendable {
    internal let tieWordEmbeddings: Bool

    internal init(_ config: GPTNeoXConfiguration) {
        self.tieWordEmbeddings = config.tieWordEmbeddings
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (checkpointKey, checkpointValue) in weights {
            var key = Self.normalizedKey(checkpointKey)
            guard !Self.shouldDrop(key) else {
                continue
            }

            if !key.hasPrefix("model.") {
                key = "model.\(key)"
            }
            if tieWordEmbeddings, Self.isOutputHead(key) {
                continue
            }
            sanitized[key] = checkpointValue
        }

        return sanitized
    }

    private static func normalizedKey(_ key: String) -> String {
        var normalized = key
        if normalized.hasPrefix("base_model.model.") {
            normalized = String(normalized.dropFirst("base_model.model.".count))
        }
        if normalized.hasPrefix("model.gpt_neox.") {
            normalized = "model." + normalized.dropFirst("model.gpt_neox.".count)
        } else if normalized.hasPrefix("gpt_neox.") {
            normalized = String(normalized.dropFirst("gpt_neox.".count))
        }
        if normalized.hasPrefix("layers.") {
            normalized = "h." + normalized.dropFirst("layers.".count)
        } else if normalized.hasPrefix("model.layers.") {
            normalized = "model.h." + normalized.dropFirst("model.layers.".count)
        }
        normalized = normalized.replacingOccurrences(of: ".layers.", with: ".h.")
        return normalized
    }

    private static func shouldDrop(_ key: String) -> Bool {
        GPTNeoXRawIgnoredSuffix.allCases.contains { key.hasSuffix($0.rawValue) }
    }

    private static func isOutputHead(_ key: String) -> Bool {
        key == "model.embed_out.weight"
            || key == "model.embed_out.scales"
            || key == "model.embed_out.biases"
    }
}

// MARK: - Layers

internal final class GPTNeoXAttention: Module {
    let layout: GPTNeoXAttentionLayout

    @ModuleInfo(key: "query_key_value") var combinedProjection: Linear
    @ModuleInfo(key: "dense") var outputProjection: Linear

    let rope: RoPE

    init(_ config: GPTNeoXConfiguration) {
        self.layout = GPTNeoXAttentionLayout(config)
        _combinedProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.combinedProjectionSize,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: true
        )
        self.rope = RoPE(
            dimensions: layout.rotaryDimensions,
            traditional: false,
            base: config.rotaryEmbeddingBase
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)
        let qkv = split(
            combinedProjection(hiddenStates),
            indices: [
                layout.queryProjectionSize,
                layout.queryProjectionSize + layout.keyValueProjectionSize
            ],
            axis: -1
        )

        var queries = qkv[0]
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = qkv[1]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = qkv[2]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
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
        .reshaped(batchSize, tokenCount, layout.queryProjectionSize)

        return outputProjection(attentionOutput)
    }
}

internal final class GPTNeoXFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "dense_h_to_4h") var upProjection: Linear
    @ModuleInfo(key: "dense_4h_to_h") var downProjection: Linear

    let activation: String

    init(_ config: GPTNeoXConfiguration) {
        self.activation = config.activationFunction.lowercased()
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.feedForwardSize,
            bias: true
        )
        _downProjection.wrappedValue = Linear(
            config.feedForwardSize,
            config.hiddenSize,
            bias: true
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

internal final class GPTNeoXTransformerBlock: Module {
    let useParallelResidual: Bool

    @ModuleInfo(key: "attention") var attention: GPTNeoXAttention
    @ModuleInfo(key: "mlp") var feedForward: GPTNeoXFeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    init(_ config: GPTNeoXConfiguration) {
        self.useParallelResidual = config.useParallelResidual
        _attention.wrappedValue = GPTNeoXAttention(config)
        _feedForward.wrappedValue = GPTNeoXFeedForward(config)
        _inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        _postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        if useParallelResidual {
            return hiddenStates
                + attention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
                + feedForward(postAttentionLayerNorm(hiddenStates))
        }

        let afterAttention = hiddenStates
            + attention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

internal final class GPTNeoXBackbone: Module {
    @ModuleInfo(key: "embed_in") var tokenEmbedding: Embedding
    @ModuleInfo(key: "embed_out") var outputHead: Linear?
    @ModuleInfo(key: "h") var layers: [GPTNeoXTransformerBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: GPTNeoXConfiguration) {
        precondition(config.vocabularySize > 0, "GPT-NeoX vocabulary size must be positive")
        precondition(config.hiddenLayers > 0, "GPT-NeoX must have at least one layer")

        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        if !config.tieWordEmbeddings {
            _outputHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            GPTNeoXTransformerBlock(config)
        }
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbedding(inputs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }

        return finalLayerNorm(hiddenStates)
    }

    func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let outputHead {
            return outputHead(hiddenStates)
        }
        return tokenEmbedding.asLinear(hiddenStates)
    }
}

internal final class GPTNeoXModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: GPTNeoXConfiguration

    @ModuleInfo(key: "model") var model: GPTNeoXBackbone

    init(_ config: GPTNeoXConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _model.wrappedValue = GPTNeoXBackbone(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
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
        GPTNeoXWeightSanitizerPlan(configuration).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        model.logits(from: hiddenStates)
    }
}

extension GPTNeoXModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["query_key_value", "dense"]) }
    }
}
