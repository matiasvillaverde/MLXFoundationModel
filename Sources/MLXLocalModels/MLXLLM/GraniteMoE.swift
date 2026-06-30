import Foundation
import MLX
import MLXNN

internal struct GraniteMoEAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ configuration: GraniteMoEConfiguration) {
        precondition(configuration.hiddenSize > 0, "hidden_size must be positive")
        precondition(configuration.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(configuration.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(
            configuration.hiddenSize.isMultiple(of: configuration.attentionHeads),
            "hidden_size must be divisible by num_attention_heads"
        )
        precondition(
            configuration.attentionHeads.isMultiple(of: configuration.kvHeads),
            "num_attention_heads must be divisible by num_key_value_heads"
        )

        self.hiddenSize = configuration.hiddenSize
        self.queryHeads = configuration.attentionHeads
        self.keyValueHeads = configuration.kvHeads
        self.headSize = configuration.hiddenSize / configuration.attentionHeads
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = configuration.attentionMultiplier
    }
}

internal struct GraniteMoERoPEPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let base: Float
    internal let scale: Float

    internal init(_ configuration: GraniteMoEConfiguration, dimensions: Int) {
        self.dimensions = dimensions
        self.base = configuration.ropeTheta
        self.scale = Self.scale(from: configuration.ropeScaling)
    }

    private static func scale(from ropeScaling: [String: StringOrNumber]?) -> Float {
        guard
            let ropeScaling,
            (ropeScaling["type"] ?? ropeScaling["rope_type"]) == .string("linear")
        else {
            return 1
        }

        guard let factor = ropeScaling["factor"]?.asFloat(), factor > 0 else {
            preconditionFailure("linear GraniteMoE rope_scaling requires a positive factor")
        }
        return 1 / factor
    }
}

internal struct GraniteMoERoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int

    internal init(_ configuration: GraniteMoEConfiguration) {
        precondition(configuration.localExperts > 0, "num_local_experts must be positive")
        precondition(configuration.expertsPerToken > 0, "num_experts_per_tok must be positive")
        precondition(
            configuration.expertsPerToken <= configuration.localExperts,
            "num_experts_per_tok cannot exceed num_local_experts"
        )

        self.expertCount = configuration.localExperts
        self.selectedExpertCount = configuration.expertsPerToken
    }

    internal func route(_ logits: MLXArray) -> (indices: MLXArray, gates: MLXArray) {
        let indices = argPartition(-logits, kth: selectedExpertCount - 1, axis: -1)[
            .ellipsis,
            ..<selectedExpertCount
        ]
        let selectedLogits = takeAlong(logits, indices, axis: -1)
        return (
            indices,
            softmax(selectedLogits.asType(.float32), axis: -1, precise: true)
        )
    }
}

internal struct GraniteMoEWeightSanitizer: Sendable {
    private let configuration: GraniteMoEConfiguration

    internal init(_ configuration: GraniteMoEConfiguration) {
        self.configuration = configuration
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights
        sanitized = sanitized.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for layerIndex in 0 ..< configuration.hiddenLayers {
            sanitized = remapPackedMoEWeights(sanitized, layerIndex: layerIndex)
        }

        return sanitized
    }

    private func remapPackedMoEWeights(
        _ weights: [String: MLXArray],
        layerIndex: Int
    ) -> [String: MLXArray] {
        var remapped = weights
        let prefix = "model.layers.\(layerIndex).block_sparse_moe"

        if let inputWeight = remapped.removeValue(forKey: "\(prefix).input_linear.weight") {
            let splitAxis = inputWeight.ndim == 3 ? 1 : 0
            let parts = inputWeight.split(parts: 2, axis: splitAxis)
            remapped["\(prefix).switch_mlp.gate_proj.weight"] = parts[0]
            remapped["\(prefix).switch_mlp.up_proj.weight"] = parts[1]
        }

        if let outputWeight = remapped.removeValue(forKey: "\(prefix).output_linear.weight") {
            remapped["\(prefix).switch_mlp.down_proj.weight"] = outputWeight
        }

        return remapped
    }
}

private final class GraniteMoERotaryEmbedding {
    private let rope: RoPE

    init(_ plan: GraniteMoERoPEPlan) {
        self.rope = RoPE(
            dimensions: plan.dimensions,
            traditional: false,
            base: plan.base,
            scale: plan.scale
        )
    }

    func apply(to input: MLXArray, offset: Int = 0) -> MLXArray {
        rope(input, offset: offset)
    }
}

private final class GraniteMoEAttention: Module {
    private let layout: GraniteMoEAttentionLayout
    private let rope: GraniteMoERotaryEmbedding

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ configuration: GraniteMoEConfiguration) {
        let layout = GraniteMoEAttentionLayout(configuration)
        self.layout = layout
        self.rope = GraniteMoERotaryEmbedding(
            GraniteMoERoPEPlan(configuration, dimensions: layout.headSize)
        )

        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: configuration.attentionBias
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: configuration.attentionBias
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: configuration.attentionBias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: configuration.attentionBias
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)

        var queries = queryProjection(input)
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope.apply(to: queries, offset: offset)
        keys = rope.apply(to: keys, offset: offset)

        return outputProjection(
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.attentionScale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, layout.queryProjectionSize)
        )
    }
}

private final class GraniteMoETopKGating: Module {
    private let routingPlan: GraniteMoERoutingPlan

    @ModuleInfo(key: "layer") private var layer: Linear

    init(inputSize: Int, routingPlan: GraniteMoERoutingPlan) {
        self.routingPlan = routingPlan
        self._layer.wrappedValue = Linear(inputSize, routingPlan.expertCount, bias: false)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (indices: MLXArray, gates: MLXArray) {
        routingPlan.route(layer(hiddenStates))
    }
}

private final class GraniteMoESparseBlock: Module, UnaryLayer {
    @ModuleInfo(key: "router") private var router: GraniteMoETopKGating
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU

    init(_ configuration: GraniteMoEConfiguration) {
        let routingPlan = GraniteMoERoutingPlan(configuration)
        self._router.wrappedValue = GraniteMoETopKGating(
            inputSize: configuration.hiddenSize,
            routingPlan: routingPlan
        )
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.intermediateSize,
            numExperts: routingPlan.expertCount,
            bias: false
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let route = router(input)
        let output = switchMLP(input, route.indices)
        let gates = route.gates.asType(output.dtype)[.ellipsis, .newAxis]
        return (output * gates).sum(axis: -2)
    }
}

private final class GraniteMoEBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: GraniteMoEAttention
    @ModuleInfo(key: "block_sparse_moe") private var sparseBlock: GraniteMoESparseBlock
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    private let residualMultiplier: Float

    init(_ configuration: GraniteMoEConfiguration) {
        self._attention.wrappedValue = GraniteMoEAttention(configuration)
        self._sparseBlock.wrappedValue = GraniteMoESparseBlock(configuration)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self.residualMultiplier = configuration.residualMultiplier
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = attention(
            inputLayerNorm(input),
            mask: mask,
            cache: cache
        )
        let hiddenStates = input + attentionOutput * residualMultiplier
        let sparseOutput = sparseBlock(postAttentionLayerNorm(hiddenStates))
        return hiddenStates + sparseOutput * residualMultiplier
    }
}

private final class GraniteMoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embeddings: Embedding
    @ModuleInfo fileprivate var layers: [GraniteMoEBlock]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    private let embeddingMultiplier: Float

    init(_ configuration: GraniteMoEConfiguration) {
        precondition(configuration.vocabularySize > 0, "vocab_size must be positive")

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        self.layers = (0 ..< configuration.hiddenLayers).map { _ in
            GraniteMoEBlock(configuration)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self.embeddingMultiplier = configuration.embeddingMultiplier
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embeddings(tokens) * embeddingMultiplier
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                mask: mask,
                cache: cache?[index]
            )
        }

        return norm(hiddenStates)
    }
}

internal final class GraniteMoEModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let configuration: GraniteMoEConfiguration
    private let logitsScaling: Float

    @ModuleInfo(key: "model") private var model: GraniteMoEBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: GraniteMoEConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self.logitsScaling = configuration.logitsScaling
        self._model.wrappedValue = GraniteMoEBackbone(configuration)

        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        greedyTokenOutput(
            logits: logits(
                from: lastTokenHiddenState(
                    model(input[text: .newAxis].tokens, cache: cache)
                )
            ),
            state: state
        )
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        GraniteMoEWeightSanitizer(configuration).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        let projected = lmHead.map { $0(hiddenStates) }
            ?? model.embeddings.asLinear(hiddenStates)
        return projected / logitsScaling
    }
}

internal struct GraniteMoEConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var logitsScaling: Float
    var attentionMultiplier: Float
    var embeddingMultiplier: Float
    var residualMultiplier: Float
    var maxPositionEmbeddings: Int
    var kvHeads: Int
    var attentionBias: Bool
    var ropeTheta: Float
    var localExperts: Int
    var expertsPerToken: Int
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "granitemoe",
        hiddenSize: Int = 1_024,
        hiddenLayers: Int = 24,
        intermediateSize: Int = 512,
        attentionHeads: Int = 16,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 49_155,
        logitsScaling: Float = 6,
        attentionMultiplier: Float = 1.0 / 64.0,
        embeddingMultiplier: Float = 12,
        residualMultiplier: Float = 0.22,
        maxPositionEmbeddings: Int = 131_072,
        kvHeads: Int? = nil,
        attentionBias: Bool = false,
        ropeTheta: Float = 1_500_000,
        localExperts: Int = 32,
        expertsPerToken: Int = 8,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.logitsScaling = logitsScaling
        self.attentionMultiplier = attentionMultiplier
        self.embeddingMultiplier = embeddingMultiplier
        self.residualMultiplier = residualMultiplier
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.kvHeads = kvHeads ?? attentionHeads
        self.attentionBias = attentionBias
        self.ropeTheta = ropeTheta
        self.localExperts = localExperts
        self.expertsPerToken = expertsPerToken
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case logitsScaling = "logits_scaling"
        case attentionMultiplier = "attention_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case residualMultiplier = "residual_multiplier"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case localExperts = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decodeIfPresent(
            Int.self,
            forKey: .attentionHeads
        ) ?? 16

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "granitemoe",
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1_024,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 512,
            attentionHeads: attentionHeads,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 49_155,
            logitsScaling: try container.decodeIfPresent(
                Float.self,
                forKey: .logitsScaling
            ) ?? 6,
            attentionMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .attentionMultiplier
            ) ?? 1.0 / 64.0,
            embeddingMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .embeddingMultiplier
            ) ?? 12,
            residualMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .residualMultiplier
            ) ?? 0.22,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? attentionHeads,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 1_500_000,
            localExperts: try container.decodeIfPresent(Int.self, forKey: .localExperts)
                ?? 32,
            expertsPerToken: try container.decodeIfPresent(Int.self, forKey: .expertsPerToken)
                ?? 8,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true
        )
    }
}

extension GraniteMoEModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
