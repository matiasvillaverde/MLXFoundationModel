// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXFast

import MLXNN

internal struct DeepseekV3Configuration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }
}

private func yarnFindCorrectionDim(
    numRotations: Float, dim: Float, base: Float = 10_000, maxPositionEmbeddings: Float = 2_048
) -> Float {
    (dim * log(maxPositionEmbeddings / (numRotations * 2 * Float.pi))) / (2 * log(base))
}

private func yarnFindCorrectionRange(
    lowRot: Float, highRot: Float, dim: Float, base: Float = 10_000,
    maxPositionEmbeddings: Float = 2_048
) -> (Float, Float) {
    let low = floor(
        yarnFindCorrectionDim(
            numRotations: lowRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings
        ))
    let high = ceil(
        yarnFindCorrectionDim(
            numRotations: highRot, dim: dim, base: base,
            maxPositionEmbeddings: maxPositionEmbeddings))
    return (max(low, 0), min(high, dim - 1))
}

private func yarnGetMScale(scale: Float = 1, mscale: Float = 1) -> Float {
    scale <= 1 ? 1.0 : 0.1 * mscale * log(scale) + 1.0
}

private func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
    let updatedMaxVal = minVal == maxVal ? maxVal + 0.001 : maxVal
    let linearFunc = (MLXArray(0 ..< dim) - minVal) / (updatedMaxVal - minVal)
    return clip(linearFunc, min: 0, max: 1)
}

private class DeepseekV3YarnRotaryEmbedding: Module {
    var mscale: Float
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let scalingFactor: Float
    let originalMaxPositionEmbeddings: Int
    let betaFast: Float
    let betaSlow: Float
    private var _freqs: MLXArray

    init(
        dim: Int,
        maxPositionEmbeddings: Int = 2_048,
        base: Float = 10_000,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4_096,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 0
    ) {
        self.mscale =
            yarnGetMScale(scale: scalingFactor, mscale: mscale)
            / yarnGetMScale(scale: scalingFactor, mscale: mscaleAllDim)
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        self.scalingFactor = scalingFactor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        let freqExtra = base ** (MLXArray(stride(from: 0, to: dim, by: 2)) / dim)
        let freqInter = scalingFactor * base ** (MLXArray(stride(from: 0, to: dim, by: 2)) / dim)
        let (low, high) = yarnFindCorrectionRange(
            lowRot: betaFast, highRot: betaSlow, dim: Float(dim), base: base,
            maxPositionEmbeddings: Float(originalMaxPositionEmbeddings))

        let freqMask = 1.0 - yarnLinearRampMask(minVal: low, maxVal: high, dim: dim / 2)

        self._freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    }

    func callAsFunction(_ input: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            self.mscale != 1.0 ? self.mscale * input : input,
            dimensions: input.shape.last ?? 0,
            traditional: true,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }
}

private func clippedSilu(_ input: MLXArray) -> MLXArray {
    clip(input * sigmoid(input), min: -100, max: 100)
}

private class DeepseekV3Attention: Module {
    var config: DeepseekV3Configuration
    var hiddenSize: Int
    var numHeads: Int
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var kvLoraRank: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var qHeadDim: Int
    var scale: Float

    let rope: DeepseekV3YarnRotaryEmbedding
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank {
            self._qAProj.wrappedValue = Linear(
                hiddenSize, qLoraRank, bias: config.attentionBias
            )
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            self._qBProj.wrappedValue = Linear(
                qLoraRank, numHeads * qHeadDim, bias: false
            )
        } else {
            self._qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank,
            numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim),
            bias: false
        )
        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

        guard let ropeScaling = config.ropeScaling,
            case .float(let scalingFactor) = ropeScaling["factor"],
            case .int(let originalMaxPositionEmbeddings) = ropeScaling[
                "original_max_position_embeddings"]
                ?? .int(4_096),
            case .float(let betaFast) = ropeScaling["beta_fast"] ?? .float(32),
            case .float(let betaSlow) = ropeScaling["beta_slow"] ?? .float(1),
            case .float(var mscale) = ropeScaling["mscale"] ?? .float(1),
            case .float(let mscaleAllDim) = ropeScaling["mscale_all_dim"] ?? .float(0)
        else {
            self.rope = DeepseekV3YarnRotaryEmbedding(dim: qkRopeHeadDim, base: ropeTheta)
            return
        }
        if mscaleAllDim != 0 {
            mscale = yarnGetMScale(scale: scalingFactor, mscale: mscaleAllDim)
            self.scale = self.scale * mscale * mscale
        }

        self.rope = DeepseekV3YarnRotaryEmbedding(
            dim: qkRopeHeadDim, maxPositionEmbeddings: maxPositionEmbeddings,
            base: ropeTheta,
            scalingFactor: scalingFactor,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
            betaFast: betaFast,
            betaSlow: betaSlow,
            mscale: mscale,
            mscaleAllDim: mscaleAllDim)
    }

    func callAsFunction(
        _ input: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batchSize, sequenceLength, _) = (input.dim(0), input.dim(1), input.dim(2))

        var queries: MLXArray
        if qLoraRank == nil {
            queries = self.qProj!(input)
        } else {
            queries = self.qBProj!(self.qALayerNorm!(self.qAProj!(input)))
        }

        queries = queries.reshaped(batchSize, sequenceLength, self.numHeads, self.qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(queries, indices: [qkNopeHeadDim], axis: -1)
        var (qNope, qPe) = (splitQ[0], splitQ[1])
        var compressedKv = self.kvAProjWithMqa(input)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(batchSize, sequenceLength, 1, self.qkRopeHeadDim).transposed(0, 2, 1, 3)
        var keyValues = self.kvBProj(kvALayerNorm(compressedKv))
        keyValues = keyValues.reshaped(batchSize, sequenceLength, self.numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(keyValues, indices: [self.qkNopeHeadDim], axis: -1)

        let (kNope, values) = (splitKv[0], splitKv[1])

        if let cache {
            qPe = self.rope(qPe, offset: cache.offset)
            kPe = self.rope(kPe, offset: cache.offset)
        } else {
            qPe = self.rope(qPe)
            kPe = self.rope(kPe)
        }
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let finalQueries = concatenated([qNope, qPe], axis: -1)
        let keys = concatenated([kNope, kPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: finalQueries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return self.oProj(output)
    }
}

private class DeepseekV3MLP: Module, UnaryLayer {
    var config: DeepseekV3Configuration
    var hiddenSize: Int
    var intermediateSize: Int
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: DeepseekV3Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.config = config
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        self.downProj(silu(self.gateProj(input)) * self.upProj(input))
    }
}

private class MoEGate: Module {
    var config: DeepseekV3Configuration
    var topK: Int?
    var normTopkProb: Bool
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var nGroup: Int
    var topkGroup: Int?

    var weight: MLXArray
    var expertScoreCorrectionBias: MLXArray

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup ?? 1
        self.topkGroup = config.topkGroup
        self.weight = zeros([self.nRoutedExperts ?? 1, config.hiddenSize])
        self.expertScoreCorrectionBias = zeros([self.nRoutedExperts ?? 1])
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        let (batchSize, sequenceLength, _) = (input.dim(0), input.dim(1), input.dim(2))

        let hiddenStates = input.matmul(weight.T)
        var scores = sigmoid(hiddenStates)
        let scoresForChoice = scores + expertScoreCorrectionBias
        let groupScores = scoresForChoice.reshaped(batchSize, sequenceLength, self.nGroup, -1)
        let topKGroup = sorted(groupScores, axis: -1)[.ellipsis, ..<2].sum(axis: -1, keepDims: true)
        var topKValue = nGroup - (topkGroup ?? 1)
        var groupIdx = argPartition(topKGroup, kth: topKValue - 1, axis: -2)[.ellipsis, ..<topKValue, 0...]
        groupIdx = broadcast(groupIdx, to: [batchSize, sequenceLength, topKValue, (nRoutedExperts ?? 1) / nGroup])
        scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)
        scores = flattened(scores, start: -2, end: -1)

        topKValue = topK ?? 1
        let inds = argPartition(-scores, kth: topKValue - 1, axis: -1)[.ellipsis, ..<topKValue]
        scores = takeAlong(scores, inds, axis: -1)
        if topK ?? 1 > 1, normTopkProb {
            let denominator = scores.sum(axis: -1, keepDims: true) + 1e-20
            scores = scores / denominator
            scores = scores * routedScalingFactor
        }

        return (inds, scores)
    }
}

private class DeepseekV3MoE: Module, UnaryLayer {
    var config: DeepseekV3Configuration
    var numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV3MLP?

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1

        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1,
            activation: clippedSilu
        )

        self.gate = MoEGate(config: config)

        if let sharedExpertCount = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * sharedExpertCount
            self._sharedExperts.wrappedValue = DeepseekV3MLP(
                config: config, intermediateSize: intermediateSize)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (indices, scores) = gate(input)
        var output = switchMLP(input, indices)
        output = (output * scores[.ellipsis, .newAxis]).sum(axis: -2)

        if let shared = sharedExperts {
            output = output + shared(input)
        }
        return output
    }
}

private class DeepseekV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV3Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV3Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV3Attention(config: config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0 {
            self.mlp = DeepseekV3MoE(config: config)
        } else {
            self.mlp = DeepseekV3MLP(config: config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ input: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let residual = selfAttn(inputLayerNorm(input), mask: mask, cache: cache)
        let hidden = input + residual
        let residual2 = mlp(postAttentionLayerNorm(hidden))
        return hidden + residual2
    }
}

private class DeepseekV3ModelInner: Module {
    var config: DeepseekV3Configuration
    var vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV3DecoderLayer]
    var startIdx: Int
    var endIdx: Int
    var numLayers: Int
    @ModuleInfo(key: "norm") var norm: RMSNorm
    var pipelineRank: Int
    var pipelineSize: Int

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.vocabSize = config.vocabSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV3DecoderLayer(config: config, layerIdx: $0)
        }
        self.startIdx = 0
        self.endIdx = layers.count
        self.numLayers = endIdx
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.pipelineRank = 0
        self.pipelineSize = 1
    }

    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embedTokens(input)

        let attentionMask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: attentionMask, cache: cache?[index])
        }

        return norm(hiddenStates)
    }
}

internal class DeepseekV3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int] = []

    var args: DeepseekV3Configuration
    fileprivate var model: DeepseekV3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DeepseekV3Configuration) {
        self.args = args
        self.model = DeepseekV3ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .block(),
            sidecarPolicy: .dropActivationScale
        ).weights

        for layerIndex in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(layerIndex)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let joined = (0 ..< (args.nRoutedExperts ?? 1)).map {
                            weights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
                    }
                }
            }
        }

        return newWeights.filter { key, _ in
            !key.starts(with: "model.layers.61") && !key.contains("rotary_emb.inv_freq")
        }
    }

    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttn, ["q_proj", "v_proj"]) }
    }
}
