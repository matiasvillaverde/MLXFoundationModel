import Foundation
import MLX
import MLXNN

// MARK: - Plans

internal struct MiniMaxAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let rotaryDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: MiniMaxConfiguration) {
        let headDimensions = config.headDim ?? (config.hiddenSize / config.attentionHeads)
        precondition(config.attentionHeads > 0, "MiniMax attention heads must be positive")
        precondition(config.kvHeads > 0, "MiniMax KV heads must be positive")
        precondition(headDimensions > 0, "MiniMax head dimensions must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "MiniMax attention heads must group KV heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = headDimensions
        self.rotaryDimensions = config.rotaryDim
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
}

internal struct MiniMaxRoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int

    internal init(_ config: MiniMaxConfiguration) {
        precondition(config.numLocalExperts > 0, "MiniMax expert count must be positive")
        precondition(config.numExpertsPerTok > 0, "MiniMax selected expert count must be positive")
        precondition(
            config.numExpertsPerTok <= config.numLocalExperts,
            "MiniMax cannot select more experts than it owns"
        )

        self.expertCount = config.numLocalExperts
        self.selectedExpertCount = config.numExpertsPerTok
    }

    internal func route(
        gates: MLXArray,
        correctionBias: MLXArray,
        outputDType: DType
    ) -> (scores: MLXArray, indices: MLXArray) {
        let originalScores = sigmoid(gates)
        let correctedScores = originalScores + correctionBias
        let indices = argPartition(
            -correctedScores,
            kth: selectedExpertCount - 1,
            axis: -1
        )[.ellipsis, ..<selectedExpertCount]

        var scores = takeAlong(originalScores, indices, axis: -1)
        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        return (scores.asType(outputDType), indices)
    }
}

private enum MiniMaxExpertProjection: CaseIterable {
    case gate
    case down
    case up

    var checkpointName: String {
        switch self {
        case .gate: "w1"
        case .down: "w2"
        case .up: "w3"
        }
    }

    var switchName: String {
        switch self {
        case .gate: "gate_proj"
        case .down: "down_proj"
        case .up: "up_proj"
        }
    }
}

internal struct MiniMaxExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: MiniMaxConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.numLocalExperts
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights
        guard packed["model.layers.0.block_sparse_moe.experts.0.w1.weight"] != nil else {
            return packed
        }

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).block_sparse_moe"
            for projection in MiniMaxExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection.checkpointName).\(tensorName)"
                    guard packed[firstKey] != nil else { continue }

                    let tensors = (0 ..< expertCount).map { expertIndex in
                        packed.removeValue(
                            forKey: "\(prefix).experts.\(expertIndex).\(projection.checkpointName).\(tensorName)"
                        )!
                    }
                    packed["\(prefix).switch_mlp.\(projection.switchName).\(tensorName)"] =
                        MLX.stacked(tensors)
                }
            }
        }

        return packed
    }
}

// MARK: - Model Components

internal final class MiniMaxAttention: Module {
    let layout: MiniMaxAttentionLayout

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ config: MiniMaxConfiguration) {
        self.layout = MiniMaxAttentionLayout(config)

        _qProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        _kProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _vProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _oProj.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: false
        )

        if config.useQkNorm {
            _qNorm.wrappedValue = RMSNorm(
                dimensions: layout.queryProjectionSize,
                eps: config.rmsNormEps
            )
            _kNorm.wrappedValue = RMSNorm(
                dimensions: layout.keyValueProjectionSize,
                eps: config.rmsNormEps
            )
        }

        self.rope = RoPE(
            dimensions: layout.rotaryDimensions,
            traditional: false,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let tokenCount = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        let values = vProj(x)

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        var queryStates = queries
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keyStates = keys
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let valueStates = values
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        queryStates = applyRotaryPosition(rope, to: queryStates, cache: cache)
        keyStates = applyRotaryPosition(rope, to: keyStates, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queryStates,
            keys: keyStates,
            values: valueStates,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return oProj(output)
    }
}

internal final class MiniMaxSparseMoeBlock: Module {
    let routingPlan: MiniMaxRoutingPlan

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: MiniMaxConfiguration) {
        self.routingPlan = MiniMaxRoutingPlan(config)

        _gate.wrappedValue = Linear(config.hiddenSize, routingPlan.expertCount, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: routingPlan.expertCount
        )
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([routingPlan.expertCount])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = routingPlan.route(
            gates: gate(x.asType(.float32)),
            correctionBias: eScoreCorrectionBias,
            outputDType: x.dtype
        )
        let expertOutput = switchMLP(x, routed.indices)
        return (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

internal final class MiniMaxDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MiniMaxAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxSparseMoeBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiniMaxConfiguration) {
        _selfAttention.wrappedValue = MiniMaxAttention(config)
        _blockSparseMoe.wrappedValue = MiniMaxSparseMoeBlock(config)
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
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hidden = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        hidden = hidden + blockSparseMoe(postAttentionLayerNorm(hidden))
        return hidden
    }
}

internal final class MiniMaxBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: MiniMaxConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in MiniMaxDecoderLayer(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (layerIndex, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hidden)
    }
}

internal final class MiniMaxModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    let configuration: MiniMaxConfiguration
    let modelType: String

    @ModuleInfo(key: "model") var model: MiniMaxBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    internal init(_ config: MiniMaxConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.modelType = config.modelType
        self._model.wrappedValue = MiniMaxBackbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
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

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { _ in KVCacheSimple() }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        sanitizedWeights = MLXQuantizedWeightSanitizer.sanitize(
            sanitizedWeights,
            strategy: .block(),
            sidecarPolicy: .dropActivationScale
        ).weights

        return MiniMaxExpertPackingPlan(configuration).pack(sanitizedWeights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}

// MARK: - Configuration

internal struct MiniMaxConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var numExpertsPerTok: Int
    var numLocalExperts: Int
    var sharedIntermediateSize: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var rotaryDim: Int
    var vocabularySize: Int
    var tieWordEmbeddings: Bool
    var scoringFunc: String
    var headDim: Int?
    var useQkNorm: Bool

    internal init(
        modelType: String = "minimax",
        hiddenSize: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        maxPositionEmbeddings: Int,
        numExpertsPerTok: Int,
        numLocalExperts: Int,
        sharedIntermediateSize: Int? = nil,
        hiddenLayers: Int,
        rmsNormEps: Float,
        ropeTheta: Float,
        rotaryDim: Int,
        vocabularySize: Int,
        tieWordEmbeddings: Bool = false,
        scoringFunc: String = "sigmoid",
        headDim: Int? = nil,
        useQkNorm: Bool = true
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numExpertsPerTok = numExpertsPerTok
        self.numLocalExperts = numLocalExperts
        self.sharedIntermediateSize = sharedIntermediateSize ?? intermediateSize
        self.hiddenLayers = hiddenLayers
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.rotaryDim = rotaryDim
        self.vocabularySize = vocabularySize
        self.tieWordEmbeddings = tieWordEmbeddings
        self.scoringFunc = scoringFunc
        self.headDim = headDim
        self.useQkNorm = useQkNorm
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numExpertsPerTok = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case sharedIntermediateSize = "shared_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scoringFunc = "scoring_func"
        case headDim = "head_dim"
        case useQkNorm = "use_qk_norm"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "minimax",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            intermediateSize: intermediateSize,
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            kvHeads: try container.decode(Int.self, forKey: .kvHeads),
            maxPositionEmbeddings: try container.decode(Int.self, forKey: .maxPositionEmbeddings),
            numExpertsPerTok: try container.decode(Int.self, forKey: .numExpertsPerTok),
            numLocalExperts: try container.decode(Int.self, forKey: .numLocalExperts),
            sharedIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .sharedIntermediateSize
            ),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            ropeTheta: try container.decode(Float.self, forKey: .ropeTheta),
            rotaryDim: try container.decode(Int.self, forKey: .rotaryDim),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            scoringFunc: try container.decodeIfPresent(String.self, forKey: .scoringFunc)
                ?? "sigmoid",
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            useQkNorm: try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        )
    }
}

// MARK: - LoRA

extension MiniMaxModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttention, ["q_proj", "v_proj"]) }
    }
}
