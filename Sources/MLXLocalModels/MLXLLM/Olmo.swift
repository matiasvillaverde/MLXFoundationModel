import Foundation
import MLX
import MLXFast
import MLXNN

internal struct OlmoAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: OlmoConfiguration) {
        precondition(config.hiddenSize > 0, "OLMo hidden size must be positive")
        precondition(config.attentionHeads > 0, "OLMo attention heads must be positive")
        precondition(config.kvHeads > 0, "OLMo KV heads must be positive")
        precondition(config.resolvedHeadDimensions > 0, "OLMo head dimensions must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "OLMo attention heads must group KV heads"
        )
        if config.headDimensions == nil {
            precondition(
                config.hiddenSize.isMultiple(of: config.attentionHeads),
                "OLMo hidden size must divide evenly across attention heads"
            )
        }

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headSize = config.resolvedHeadDimensions
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

internal struct OlmoWeightSanitizerPlan: Equatable, Sendable {
    internal let tieWordEmbeddings: Bool

    internal init(_ config: OlmoConfiguration) {
        self.tieWordEmbeddings = config.tieWordEmbeddings
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

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

        splitLegacyPackedAttentionWeights(in: &sanitized)
        splitLegacyPackedFeedForwardWeights(in: &sanitized)
        return sanitized
    }

    private static func normalizedKey(_ key: String) -> String {
        var normalized = key
        if normalized.hasPrefix("base_model.model.") {
            normalized = String(normalized.dropFirst("base_model.model.".count))
        }
        if normalized.hasPrefix("model.transformer.") {
            normalized = normalized
                .replacingOccurrences(of: "model.transformer.wte.", with: "model.embed_tokens.")
                .replacingOccurrences(of: "model.transformer.blocks.", with: "model.layers.")
                .replacingOccurrences(of: "model.transformer.ff_out.", with: "lm_head.")
        }
        if normalized.hasPrefix("model.lm_head.") {
            normalized = String(normalized.dropFirst("model.".count))
        }
        normalized = normalized
            .replacingOccurrences(of: ".att_norm.", with: ".input_layernorm.")
            .replacingOccurrences(of: ".ff_norm.", with: ".post_attention_layernorm.")
            .replacingOccurrences(of: ".attn_out.", with: ".self_attn.o_proj.")
            .replacingOccurrences(of: ".att_proj.", with: ".self_attn.att_proj.")
            .replacingOccurrences(of: ".ff_out.", with: ".mlp.down_proj.")
            .replacingOccurrences(of: ".ff_proj.", with: ".mlp.ff_proj.")
        return normalized
    }

    private static func shouldDrop(_ key: String) -> Bool {
        key.contains("rotary_emb.inv_freq")
            || key.hasSuffix(".input_layernorm.weight")
            || key.hasSuffix(".input_layernorm.bias")
            || key.hasSuffix(".post_attention_layernorm.weight")
            || key.hasSuffix(".post_attention_layernorm.bias")
            || key == "model.norm.weight"
            || key == "model.norm.bias"
            || key == "model.transformer.norm.weight"
            || key == "model.transformer.norm.bias"
    }

    private static func isOutputHead(_ key: String) -> Bool {
        key == "lm_head.weight"
            || key == "lm_head.scales"
            || key == "lm_head.biases"
            || key == "model.lm_head.weight"
            || key == "model.lm_head.scales"
            || key == "model.lm_head.biases"
    }

    private func splitLegacyPackedAttentionWeights(in weights: inout [String: MLXArray]) {
        for key in weights.keys.sorted() where key.hasSuffix(".self_attn.att_proj.weight") {
            let prefix = String(key.dropLast(".att_proj.weight".count))
            guard let packed = weights.removeValue(forKey: key) else {
                continue
            }
            let splitWeights = split(packed, parts: 3, axis: 0)
            weights["\(prefix).q_proj.weight"] = splitWeights[0]
            weights["\(prefix).k_proj.weight"] = splitWeights[1]
            weights["\(prefix).v_proj.weight"] = splitWeights[2]
        }

        for key in weights.keys.sorted() where key.hasSuffix(".self_attn.att_proj.bias") {
            let prefix = String(key.dropLast(".att_proj.bias".count))
            guard let packed = weights.removeValue(forKey: key) else {
                continue
            }
            let splitBiases = split(packed, parts: 3, axis: 0)
            weights["\(prefix).q_proj.bias"] = splitBiases[0]
            weights["\(prefix).k_proj.bias"] = splitBiases[1]
            weights["\(prefix).v_proj.bias"] = splitBiases[2]
        }
    }

    private func splitLegacyPackedFeedForwardWeights(in weights: inout [String: MLXArray]) {
        for key in weights.keys.sorted() where key.hasSuffix(".mlp.ff_proj.weight") {
            let prefix = String(key.dropLast(".ff_proj.weight".count))
            guard let packed = weights.removeValue(forKey: key) else {
                continue
            }
            let splitWeights = split(packed, parts: 2, axis: 0)
            weights["\(prefix).up_proj.weight"] = splitWeights[0]
            weights["\(prefix).gate_proj.weight"] = splitWeights[1]
        }

        for key in weights.keys.sorted() where key.hasSuffix(".mlp.ff_proj.bias") {
            let prefix = String(key.dropLast(".ff_proj.bias".count))
            guard let packed = weights.removeValue(forKey: key) else {
                continue
            }
            let splitBiases = split(packed, parts: 2, axis: 0)
            weights["\(prefix).up_proj.bias"] = splitBiases[0]
            weights["\(prefix).gate_proj.bias"] = splitBiases[1]
        }
    }
}

private final class OlmoAttention: Module {
    private let layout: OlmoAttentionLayout
    private let rope: RoPELayer
    private let clipQKV: Float?

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: OlmoConfiguration) {
        let layout = OlmoAttentionLayout(config)
        self.layout = layout
        self.rope = initializeRope(
            dims: layout.headSize,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        self.clipQKV = config.clipQKV

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

        var queries = clipped(queryProjection(hiddenStates))
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = clipped(keyProjection(hiddenStates))
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = clipped(valueProjection(hiddenStates))
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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

    private func clipped(_ array: MLXArray) -> MLXArray {
        guard let clipQKV else {
            return array
        }
        return clip(array, min: -clipQKV, max: clipQKV)
    }
}

private final class OlmoFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: OlmoConfiguration) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class OlmoBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: OlmoAttention
    @ModuleInfo(key: "mlp") private var feedForward: OlmoFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: LayerNorm

    init(_ config: OlmoConfiguration) {
        self._attention.wrappedValue = OlmoAttention(config)
        self._feedForward.wrappedValue = OlmoFeedForward(config)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps,
            affine: false
        )
        self._postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps,
            affine: false
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

private final class OlmoBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [OlmoBlock]
    @ModuleInfo(key: "norm") private var finalNorm: LayerNorm

    init(_ config: OlmoConfiguration) {
        precondition(config.embeddingSize > 0, "OLMo embedding size must be positive")
        precondition(config.hiddenLayers > 0, "OLMo must have at least one layer")

        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.embeddingSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            OlmoBlock(config)
        }
        self._finalNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps,
            affine: false
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        let mask = createAttentionMask(h: hiddenStates, cache: cache?.first)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

internal final class OlmoModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: OlmoBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    private let config: OlmoConfiguration

    internal init(_ config: OlmoConfiguration) {
        self.config = config
        self.vocabularySize = config.embeddingSize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self._model.wrappedValue = OlmoBackbone(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.embeddingSize,
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
        OlmoWeightSanitizerPlan(config).sanitize(weights)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { _ in KVCacheSimple() }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

internal struct OlmoConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var layerNormEps: Float
    var vocabularySize: Int
    var embeddingSize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var mlpBias: Bool
    var hiddenActivation: String
    var clipQKV: Float?

    internal var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    internal init(
        modelType: String = "olmo",
        hiddenSize: Int = 2_048,
        hiddenLayers: Int = 16,
        intermediateSize: Int = 8_192,
        attentionHeads: Int = 16,
        headDimensions: Int? = nil,
        layerNormEps: Float = 1e-5,
        vocabularySize: Int = 50_304,
        embeddingSize: Int? = nil,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int = 2_048,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        hiddenActivation: String = "silu",
        clipQKV: Float? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.layerNormEps = layerNormEps
        self.vocabularySize = vocabularySize
        self.embeddingSize = embeddingSize ?? vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.hiddenActivation = hiddenActivation
        self.clipQKV = clipQKV
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case layerNormEps = "layer_norm_eps"
        case vocabularySize = "vocab_size"
        case embeddingSize = "embedding_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case hiddenActivation = "hidden_act"
        case clipQKV = "clip_qkv"
    }

    enum LegacyCodingKeys: String, CodingKey {
        case hiddenSize = "d_model"
        case hiddenLayers = "n_layers"
        case packedIntermediateSize = "mlp_hidden_size"
        case mlpRatio = "mlp_ratio"
        case attentionHeads = "n_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case weightTying = "weight_tying"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let legacy = try decoder.container(keyedBy: LegacyCodingKeys.self)
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? legacy.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? 2_048
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
            ?? legacy.decodeIfPresent(Int.self, forKey: .hiddenLayers)
            ?? 16
        let attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? legacy.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? 16
        let vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? 50_304
        let legacyMLPRatio = try legacy.decodeIfPresent(Int.self, forKey: .mlpRatio)
            ?? 4
        let intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? legacy.decodeIfPresent(Int.self, forKey: .packedIntermediateSize).map {
                $0 / 2
            }
            ?? hiddenSize * legacyMLPRatio
        let ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeScaling
        ) ?? legacy.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeParameters
        )

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "olmo",
            hiddenSize: hiddenSize,
            hiddenLayers: hiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: attentionHeads,
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
                ?? legacy.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            vocabularySize: vocabularySize,
            embeddingSize: try container.decodeIfPresent(Int.self, forKey: .embeddingSize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 2_048,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(
                Bool.self,
                forKey: .ropeTraditional
            ) ?? false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? legacy.decodeIfPresent(Bool.self, forKey: .weightTying)
                ?? false,
            attentionBias: try container.decodeIfPresent(
                Bool.self,
                forKey: .attentionBias
            ) ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            hiddenActivation: try container.decodeIfPresent(
                String.self,
                forKey: .hiddenActivation
            ) ?? "silu",
            clipQKV: try container.decodeIfPresent(Float.self, forKey: .clipQKV)
        )
    }
}

extension OlmoModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
