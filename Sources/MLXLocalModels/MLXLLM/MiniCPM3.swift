import Foundation
import MLX
import MLXFast
import MLXNN

internal struct MiniCPM3AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let queryLowRank: Int
    internal let keyValueLowRank: Int
    internal let nopeHeadSize: Int
    internal let ropeHeadSize: Int
    internal let valueHeadSize: Int
    internal let queryHeadSize: Int
    internal let queryProjectionSize: Int
    internal let compressedKeyValueSize: Int
    internal let keyValueProjectionSize: Int
    internal let outputProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: MiniCPM3Configuration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.qLoraRank > 0, "q_lora_rank must be positive")
        precondition(config.kvLoraRank > 0, "kv_lora_rank must be positive")
        precondition(config.qkNopeHeadDim > 0, "qk_nope_head_dim must be positive")
        precondition(config.qkRopeHeadDim > 0, "qk_rope_head_dim must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "hidden_size must divide evenly across attention heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.queryLowRank = config.qLoraRank
        self.keyValueLowRank = config.kvLoraRank
        self.nopeHeadSize = config.qkNopeHeadDim
        self.ropeHeadSize = config.qkRopeHeadDim
        self.valueHeadSize = config.hiddenSize / config.attentionHeads
        self.queryHeadSize = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.queryProjectionSize = config.attentionHeads * queryHeadSize
        self.compressedKeyValueSize = config.kvLoraRank + config.qkRopeHeadDim
        self.keyValueProjectionSize = config.attentionHeads * (
            config.qkNopeHeadDim + valueHeadSize
        )
        self.outputProjectionSize = config.attentionHeads * valueHeadSize
        self.attentionScale = powf(Float(queryHeadSize), -0.5)
    }
}

internal struct MiniCPM3ScalePlan: Equatable, Sendable {
    internal let embeddingScale: Float
    internal let residualScale: Float
    internal let logitDivisor: Float

    internal init(_ config: MiniCPM3Configuration) {
        precondition(config.hiddenLayers > 0, "num_hidden_layers must be positive")
        precondition(config.dimModelBase > 0, "dim_model_base must be positive")

        self.embeddingScale = config.scaleEmb
        self.residualScale = config.scaleDepth / Float(config.hiddenLayers).squareRoot()
        self.logitDivisor = Float(config.hiddenSize) / Float(config.dimModelBase)
    }
}

private final class MiniCPM3Attention: Module {
    fileprivate let layout: MiniCPM3AttentionLayout
    private let rope: MiniCPM3RotaryEmbedding

    @ModuleInfo(key: "q_a_proj") private var queryLowRankProjection: Linear
    @ModuleInfo(key: "q_a_layernorm") private var queryLowRankNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") private var queryProjection: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") private var keyValueLowRankProjection: Linear
    @ModuleInfo(key: "kv_a_layernorm") private var keyValueLowRankNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") private var keyValueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: MiniCPM3Configuration) {
        let layout = MiniCPM3AttentionLayout(config)
        self.layout = layout
        self.rope = MiniCPM3RotaryEmbedding(config, dimensions: layout.ropeHeadSize)

        _queryLowRankProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryLowRank,
            bias: config.attentionBias
        )
        _queryLowRankNorm.wrappedValue = RMSNorm(dimensions: layout.queryLowRank)
        _queryProjection.wrappedValue = Linear(
            layout.queryLowRank,
            layout.queryProjectionSize,
            bias: false
        )
        _keyValueLowRankProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.compressedKeyValueSize,
            bias: config.attentionBias
        )
        _keyValueLowRankNorm.wrappedValue = RMSNorm(dimensions: layout.keyValueLowRank)
        _keyValueProjection.wrappedValue = Linear(
            layout.keyValueLowRank,
            layout.keyValueProjectionSize,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.outputProjectionSize,
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

        let projectedQueries = queryProjection(queryLowRankNorm(queryLowRankProjection(hiddenStates)))
        let queries = projectedQueries
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.queryHeadSize)
            .transposed(0, 2, 1, 3)
        let queryParts = split(queries, indices: [layout.nopeHeadSize], axis: -1)
        let queryNope = queryParts[0]
        var queryRope = queryParts[1]

        let compressedKeyValues = keyValueLowRankProjection(hiddenStates)
        let compressedParts = split(
            compressedKeyValues,
            indices: [layout.keyValueLowRank],
            axis: -1
        )
        let latentKeyValues = compressedParts[0]
        var keyRope = compressedParts[1]
            .reshaped(batchSize, tokenCount, 1, layout.ropeHeadSize)
            .transposed(0, 2, 1, 3)

        let projectedKeyValues = keyValueProjection(keyValueLowRankNorm(latentKeyValues))
            .reshaped(batchSize, tokenCount, layout.attentionHeads, -1)
            .transposed(0, 2, 1, 3)
        let keyValueParts = split(projectedKeyValues, indices: [layout.nopeHeadSize], axis: -1)

        let offset = cache?.offset ?? 0
        queryRope = rope(queryRope, offset: offset)
        keyRope = rope(keyRope, offset: offset)

        let keys = concatenated(
            [
                keyValueParts[0],
                repeated(keyRope, count: layout.attentionHeads, axis: 1),
            ],
            axis: -1
        )
        let values = keyValueParts[1]
        let finalQueries = concatenated([queryNope, queryRope], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: finalQueries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(output)
    }
}

private final class MiniCPM3RotaryEmbedding {
    private let dimensions: Int
    private let base: Float
    private let traditional: Bool
    private let longRoPEPlan: SuScaledRoPEPlan?
    private let shortFrequencies: MLXArray?
    private let longFrequencies: MLXArray?

    init(_ config: MiniCPM3Configuration, dimensions: Int) {
        self.dimensions = dimensions
        self.base = config.ropeTheta
        self.traditional = config.ropeTraditional

        let ropeType = Self.ropeType(config.ropeScaling)
        if ropeType == "longrope" {
            let plan = Self.longRoPEPlan(
                config.ropeScaling,
                dimensions: dimensions,
                base: config.ropeTheta,
                maxPositionEmbeddings: config.maxPositionEmbeddings
            )
            self.longRoPEPlan = plan
            self.shortFrequencies = MLXArray(
                plan.frequencyValues(useLongFrequencies: false)
            ).asType(.float32)
            self.longFrequencies = MLXArray(
                plan.frequencyValues(useLongFrequencies: true)
            ).asType(.float32)
        } else {
            precondition(
                ropeType == nil,
                "MiniCPM3 only supports default RoPE and longrope scaling"
            )
            self.longRoPEPlan = nil
            self.shortFrequencies = nil
            self.longFrequencies = nil
        }
    }

    func callAsFunction(_ input: MLXArray, offset: Int) -> MLXArray {
        guard let plan = longRoPEPlan else {
            return MLXFast.RoPE(
                input,
                dimensions: dimensions,
                traditional: traditional,
                base: base,
                scale: 1,
                offset: offset
            )
        }

        let positionLimit = offset + input.dim(-2)
        let scaledInput = scaledLongRoPEInput(input, plan: plan, positionLimit: positionLimit)
        return MLXFast.RoPE(
            scaledInput,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1,
            offset: offset,
            freqs: longRoPEFrequencies(plan: plan, positionLimit: positionLimit)
        )
    }

    private func longRoPEFrequencies(
        plan: SuScaledRoPEPlan,
        positionLimit: Int
    ) -> MLXArray {
        plan.usesLongFrequencies(positionLimit: positionLimit)
            ? longFrequencies!
            : shortFrequencies!
    }

    private func scaledLongRoPEInput(
        _ input: MLXArray,
        plan: SuScaledRoPEPlan,
        positionLimit: Int
    ) -> MLXArray {
        precondition(
            input.dim(-1) >= dimensions,
            "input head dimension must contain the rotated dimensions"
        )

        let scale = plan.scale(positionLimit: positionLimit)
        guard scale != 1 else { return input }

        if input.dim(-1) == dimensions {
            return input * scale
        }

        return concatenated(
            [
                input[.ellipsis, 0 ..< dimensions] * scale,
                input[.ellipsis, dimensions...]
            ],
            axis: -1
        )
    }

    private static func ropeType(_ config: [String: StringOrNumber]?) -> String? {
        guard let value = config?["type"] ?? config?["rope_type"],
              case .string(let type) = value else {
            return nil
        }
        return type
    }

    private static func longRoPEPlan(
        _ config: [String: StringOrNumber]?,
        dimensions: Int,
        base: Float,
        maxPositionEmbeddings: Int
    ) -> SuScaledRoPEPlan {
        guard let config else {
            preconditionFailure("longrope requires rope_scaling")
        }
        guard let originalMaxPositionEmbeddings =
            config["original_max_position_embeddings"]?.asInt() else {
            preconditionFailure("longrope requires original_max_position_embeddings")
        }
        guard let shortFactor = config["short_factor"]?.asFloats() else {
            preconditionFailure("longrope requires short_factor")
        }
        guard let longFactor = config["long_factor"]?.asFloats() else {
            preconditionFailure("longrope requires long_factor")
        }

        return SuScaledRoPEPlan(
            dimensions: dimensions,
            base: base,
            maxPositionEmbeddings: maxPositionEmbeddings,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
            shortFactor: shortFactor,
            longFactor: longFactor
        )
    }
}

private final class MiniCPM3FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: MiniCPM3Configuration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(hiddenStates)) * upProjection(hiddenStates))
    }
}

private final class MiniCPM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: MiniCPM3Attention
    @ModuleInfo(key: "mlp") private var feedForward: MiniCPM3FeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    private let residualScale: Float

    init(_ config: MiniCPM3Configuration, scalePlan: MiniCPM3ScalePlan) {
        self.residualScale = scalePlan.residualScale
        _selfAttention.wrappedValue = MiniCPM3Attention(config)
        _feedForward.wrappedValue = MiniCPM3FeedForward(config)
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
            + selfAttention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
            * residualScale
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
            * residualScale
    }
}

private final class MiniCPM3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo fileprivate var layers: [MiniCPM3DecoderLayer]
    @ModuleInfo(key: "norm") private var finalNorm: RMSNorm

    private let scalePlan: MiniCPM3ScalePlan

    init(_ config: MiniCPM3Configuration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        let scalePlan = MiniCPM3ScalePlan(config)
        self.scalePlan = scalePlan
        _tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        layers = (0 ..< config.hiddenLayers).map { _ in
            MiniCPM3DecoderLayer(config, scalePlan: scalePlan)
        }
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = tokenEmbeddings(tokens)
        if scalePlan.embeddingScale != 1 {
            hiddenStates = hiddenStates * scalePlan.embeddingScale
        }

        let mask = createAttentionMask(h: hiddenStates, cache: cache)
        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalNorm(hiddenStates)
    }
}

internal final class MiniCPM3Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") fileprivate var model: MiniCPM3Backbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    private let scalePlan: MiniCPM3ScalePlan
    private let tieWordEmbeddings: Bool

    internal init(_ config: MiniCPM3Configuration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.attentionHeads, count: config.hiddenLayers)
        self.scalePlan = MiniCPM3ScalePlan(config)
        self.tieWordEmbeddings = config.tieWordEmbeddings
        _model.wrappedValue = MiniCPM3Backbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    internal func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(tokens, cache: cache))
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
        weights.filter { key, _ in
            if key.contains("self_attn.rotary_emb.inv_freq") { return false }
            if tieWordEmbeddings, key == "lm_head.weight" { return false }
            return true
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            let scaledHiddenStates = scalePlan.logitDivisor == 1
                ? hiddenStates
                : hiddenStates / scalePlan.logitDivisor
            return lmHead(scaledHiddenStates)
        }
        return model.tokenEmbeddings.asLinear(hiddenStates)
    }
}

internal struct MiniCPM3Configuration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var dimModelBase: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var qLoraRank: Int
    var qkNopeHeadDim: Int
    var qkRopeHeadDim: Int
    var kvLoraRank: Int
    var scaleDepth: Float
    var scaleEmb: Float
    var maxPositionEmbeddings: Int
    var attentionBias: Bool
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "minicpm3",
        hiddenSize: Int,
        dimModelBase: Int? = nil,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        qLoraRank: Int,
        qkNopeHeadDim: Int,
        qkRopeHeadDim: Int,
        kvLoraRank: Int,
        scaleDepth: Float = 1,
        scaleEmb: Float = 1,
        maxPositionEmbeddings: Int,
        attentionBias: Bool = false,
        ropeTheta: Float = 1_000_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.dimModelBase = dimModelBase ?? hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.qLoraRank = qLoraRank
        self.qkNopeHeadDim = qkNopeHeadDim
        self.qkRopeHeadDim = qkRopeHeadDim
        self.kvLoraRank = kvLoraRank
        self.scaleDepth = scaleDepth
        self.scaleEmb = scaleEmb
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case dimModelBase = "dim_model_base"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case qLoraRank = "q_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case kvLoraRank = "kv_lora_rank"
        case scaleDepth = "scale_depth"
        case scaleEmb = "scale_emb"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "minicpm3",
            hiddenSize: hiddenSize,
            dimModelBase: try container.decodeIfPresent(Int.self, forKey: .dimModelBase)
                ?? hiddenSize,
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            qLoraRank: try container.decode(Int.self, forKey: .qLoraRank),
            qkNopeHeadDim: try container.decode(Int.self, forKey: .qkNopeHeadDim),
            qkRopeHeadDim: try container.decode(Int.self, forKey: .qkRopeHeadDim),
            kvLoraRank: try container.decode(Int.self, forKey: .kvLoraRank),
            scaleDepth: try container.decodeIfPresent(Float.self, forKey: .scaleDepth) ?? 1,
            scaleEmb: try container.decodeIfPresent(Float.self, forKey: .scaleEmb) ?? 1,
            maxPositionEmbeddings: try container.decode(Int.self, forKey: .maxPositionEmbeddings),
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 1_000_000,
            ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                ?? false,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

extension MiniCPM3Model: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map {
            ($0.selfAttention, ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"])
        }
    }
}
