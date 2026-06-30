import Foundation
import MLX
import MLXFast
import MLXNN

internal struct Phi3SmallAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let queryHeadsPerKeyValueHead: Int
    internal let headDimensions: Int
    internal let packedProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: Phi3SmallConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.keyValueHeads > 0, "num_key_value_heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "hidden_size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.keyValueHeads),
            "attention heads must group key-value heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.queryHeadsPerKeyValueHead = config.attentionHeads / config.keyValueHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.packedProjectionSize = (config.attentionHeads + 2 * config.keyValueHeads)
            * headDimensions

        if config.mupUseScaling {
            precondition(config.mupAttentionMultiplier > 0, "mup_attn_multiplier must be positive")
            self.attentionScale = 1 / (Float(headDimensions) / config.mupAttentionMultiplier)
        } else {
            self.attentionScale = pow(Float(headDimensions), -0.5)
        }
    }
}

internal struct Phi3SmallBlockSparsePlan: Equatable, Sendable {
    internal let blockSize: Int
    internal let localBlockCount: Int
    internal let verticalStride: Int
    internal let keyValueHeads: Int
    internal let queryHeadsPerKeyValueHead: Int

    internal init?(_ config: Phi3SmallConfiguration, layout: Phi3SmallAttentionLayout, layerIndex: Int) {
        precondition(
            config.denseAttentionEveryN > 0,
            "dense_attention_every_n_layers must be positive"
        )
        guard layerIndex.isMultiple(of: config.denseAttentionEveryN) else {
            return nil
        }
        precondition(
            config.blockSparseBlockSize == 32 || config.blockSparseBlockSize == 64,
            "blocksparse_block_size must be 32 or 64"
        )
        precondition(config.blockSparseLocalBlocks > 0, "blocksparse_num_local_blocks must be positive")
        precondition(config.blockSparseVerticalStride > 0, "blocksparse_vert_stride must be positive")

        self.blockSize = config.blockSparseBlockSize
        self.localBlockCount = config.blockSparseLocalBlocks
        self.verticalStride = config.blockSparseVerticalStride
        self.keyValueHeads = layout.keyValueHeads
        self.queryHeadsPerKeyValueHead = layout.queryHeadsPerKeyValueHead
    }

    internal func denseMask(queryLength: Int, keyLength: Int) -> MLXArray {
        let keyBlockCount = (keyLength + blockSize - 1) / blockSize
        let queryBlockCount = (queryLength + blockSize - 1) / blockSize
        let firstQueryBlock = keyBlockCount - queryBlockCount
        let headCount = keyValueHeads * queryHeadsPerKeyValueHead
        var values = [Bool](
            repeating: false,
            count: keyValueHeads * queryHeadsPerKeyValueHead * queryLength * keyLength
        )

        for keyValueHead in 0 ..< keyValueHeads {
            for queryHeadOffset in 0 ..< queryHeadsPerKeyValueHead {
                let head = keyValueHead * queryHeadsPerKeyValueHead + queryHeadOffset
                for queryIndex in 0 ..< queryLength {
                    let queryBlock = firstQueryBlock + queryIndex / blockSize
                    for keyIndex in 0 ..< keyLength {
                        let keyBlock = keyIndex / blockSize
                        let vertical = (keyBlock + head + 1).isMultiple(of: verticalStride)
                        let allowed = queryBlock >= keyBlock
                            && (queryBlock - keyBlock < localBlockCount || vertical)
                        let flatIndex = (((keyValueHead * queryHeadsPerKeyValueHead + queryHeadOffset)
                            * queryLength + queryIndex) * keyLength) + keyIndex
                        values[flatIndex] = allowed
                    }
                }
            }
        }

        precondition(headCount == keyValueHeads * queryHeadsPerKeyValueHead)
        return MLXArray(values)
            .reshaped(keyValueHeads, queryHeadsPerKeyValueHead, queryLength, keyLength)
    }
}

private func phi3SmallGeGELU(_ input: MLXArray, limit: Float) -> MLXArray {
    var geluInput = input[.ellipsis, .stride(by: 2)]
    var linearInput = input[.ellipsis, .stride(from: 1, by: 2)]

    geluInput = MLX.where(
        isInf(geluInput),
        geluInput,
        clip(geluInput, max: MLXArray(limit))
    )
    linearInput = MLX.where(
        isInf(linearInput),
        linearInput,
        clip(linearInput, min: MLXArray(-limit), max: MLXArray(limit))
    )

    let activated = geluInput * sigmoid(1.702 * geluInput)
    return activated * (linearInput + 1)
}

private final class Phi3SmallAttention: Module {
    private let layout: Phi3SmallAttentionLayout
    private let blockSparsePlan: Phi3SmallBlockSparsePlan?
    private let rope: RoPE

    @ModuleInfo(key: "query_key_value") private var queryKeyValue: Linear
    @ModuleInfo(key: "dense") private var dense: Linear

    init(_ config: Phi3SmallConfiguration, layerIndex: Int) {
        let layout = Phi3SmallAttentionLayout(config)
        self.layout = layout
        self.blockSparsePlan = Phi3SmallBlockSparsePlan(config, layout: layout, layerIndex: layerIndex)
        self.rope = RoPE(
            dimensions: layout.headDimensions,
            traditional: false,
            base: config.ropeBase,
            scale: config.ropeScale
        )
        _queryKeyValue.wrappedValue = Linear(
            config.hiddenSize,
            layout.packedProjectionSize,
            bias: true
        )
        _dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)
        let qkv = queryKeyValue(input).reshaped(
            batchSize,
            tokenCount,
            layout.keyValueHeads,
            layout.queryHeadsPerKeyValueHead + 2,
            layout.headDimensions
        )

        var queries = qkv[.ellipsis, ..<layout.queryHeadsPerKeyValueHead, 0...]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = qkv[.ellipsis, layout.queryHeadsPerKeyValueHead, 0...]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var values = qkv[.ellipsis, layout.queryHeadsPerKeyValueHead + 1, 0...]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output: MLXArray
        if let blockSparsePlan {
            if let cache {
                let updated = updateCacheReturningMaterializedKV(
                    keys: keys,
                    values: values,
                    cache: cache
                )
                keys = updated.keys
                values = updated.values
            }
            output = blockSparseAttention(
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                plan: blockSparsePlan
            )
        } else {
            output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.attentionScale,
                mask: mask
            )
        }

        return dense(
            output
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, tokenCount, -1)
        )
    }

    private func blockSparseAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        plan: Phi3SmallBlockSparsePlan
    ) -> MLXArray {
        let batchSize = queries.dim(0)
        let queryLength = queries.dim(2)
        let keyLength = keys.dim(2)
        let groupedQueries = (queries * layout.attentionScale)
            .reshaped(
                batchSize,
                layout.keyValueHeads,
                layout.queryHeadsPerKeyValueHead,
                queryLength,
                layout.headDimensions
            )
        let expandedKeys = keys[.ellipsis, .newAxis, 0..., 0...]
        let expandedValues = values[.ellipsis, .newAxis, 0..., 0...]

        var scores = matmul(groupedQueries, expandedKeys.transposed(0, 1, 2, 4, 3))
        if let maskArray = mask.mask {
            scores = scores + maskArray
        }

        let rawSparseMask = plan.denseMask(queryLength: queryLength, keyLength: keyLength)
        let sparseMask = rawSparseMask[.newAxis, 0..., 0..., 0..., 0...]
        scores = scores + MLX.where(
            sparseMask,
            MLXArray(0, dtype: scores.dtype),
            MLXArray(-Float.infinity, dtype: scores.dtype)
        )
        let probabilities = softmax(scores, axis: -1, precise: true)
        return matmul(probabilities, expandedValues)
            .reshaped(batchSize, layout.attentionHeads, queryLength, layout.headDimensions)
    }
}

private final class Phi3SmallFeedForward: Module, UnaryLayer {
    private let gegeluLimit: Float

    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: Phi3SmallConfiguration) {
        self.gegeluLimit = config.gegeluLimit
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            2 * config.intermediateSize,
            bias: true
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: true
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(phi3SmallGeGELU(upProjection(input), limit: gegeluLimit))
    }
}

private final class Phi3SmallBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Phi3SmallAttention
    @ModuleInfo(key: "mlp") private var feedForward: Phi3SmallFeedForward
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: LayerNorm

    init(_ config: Phi3SmallConfiguration, layerIndex: Int) {
        _attention.wrappedValue = Phi3SmallAttention(config, layerIndex: layerIndex)
        _feedForward.wrappedValue = Phi3SmallFeedForward(config)
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
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = input + attention(inputLayerNorm(input), mask: mask, cache: cache)
        return attended + feedForward(postAttentionLayerNorm(attended))
    }
}

private final class Phi3SmallBackbone: Module {
    private let embeddingMultiplier: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Phi3SmallBlock]
    @ModuleInfo(key: "final_layernorm") private var finalLayerNorm: LayerNorm

    init(_ config: Phi3SmallConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")
        self.embeddingMultiplier = config.mupEmbeddingMultiplier
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map {
            Phi3SmallBlock(config, layerIndex: $0)
        }
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        if embeddingMultiplier != 0 {
            hidden = (hidden * embeddingMultiplier).asType(hidden.dtype)
        }

        let mask = createAttentionMask(h: hidden, cache: cache, returnArray: true)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return finalLayerNorm(hidden)
    }
}

internal final class Phi3SmallModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let widthMultiplier: Float
    private let dummyTokenIDs: [Int]

    @ModuleInfo(key: "model") private var model: Phi3SmallBackbone

    init(_ config: Phi3SmallConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.keyValueHeads, count: config.hiddenLayers)
        self.widthMultiplier = config.mupWidthMultiplier
        self.dummyTokenIDs = Self.dummySuppressedTokenIDs.filter { $0 < config.vocabularySize }
        _model.wrappedValue = Phi3SmallBackbone(config)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hidden = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hidden), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        var logits = model.embedTokens.asLinear(hiddenStates)
        if widthMultiplier != 0 {
            logits = logits / widthMultiplier
        }
        return suppressDummyTokens(in: logits)
    }

    private func suppressDummyTokens(in logits: MLXArray) -> MLXArray {
        guard !dummyTokenIDs.isEmpty else {
            return logits
        }
        let indices = MLXArray(dummyTokenIDs).asType(.uint32)
        let bias = MLXArray.zeros([vocabularySize], type: Float32.self)
            .at[indices]
            .add(MLXArray(-Float.infinity))
            .asType(logits.dtype)
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[shape.count - 1] = vocabularySize
        return logits + bias.reshaped(shape)
    }

    private static let dummySuppressedTokenIDs: [Int] = [
        100_256,
        100_258,
        100_259,
        100_260,
        100_264,
        100_265
    ] + Array(100_267 ..< 100_352)
}

internal struct Phi3SmallConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var hiddenSize: Int
    internal var denseAttentionEveryN: Int
    internal var intermediateSize: Int
    internal var gegeluLimit: Float
    internal var hiddenLayers: Int
    internal var attentionHeads: Int
    internal var layerNormEps: Float
    internal var vocabularySize: Int
    internal var keyValueHeads: Int
    internal var mupAttentionMultiplier: Float
    internal var mupUseScaling: Bool
    internal var mupEmbeddingMultiplier: Float
    internal var mupWidthMultiplier: Float
    internal var ropeBase: Float
    internal var ropeScale: Float
    internal var blockSparseBlockSize: Int
    internal var blockSparseLocalBlocks: Int
    internal var blockSparseVerticalStride: Int

    internal init(
        modelType: String = "phi3small",
        hiddenSize: Int = 4_096,
        denseAttentionEveryN: Int = 2,
        intermediateSize: Int = 14_336,
        gegeluLimit: Float = 20,
        hiddenLayers: Int = 32,
        attentionHeads: Int = 32,
        layerNormEps: Float = 1e-5,
        vocabularySize: Int = 100_352,
        keyValueHeads: Int = 8,
        mupAttentionMultiplier: Float = 1,
        mupUseScaling: Bool = true,
        mupEmbeddingMultiplier: Float = 10,
        mupWidthMultiplier: Float = 8,
        ropeBase: Float = 1_000_000,
        ropeScale: Float = 1,
        blockSparseBlockSize: Int = 64,
        blockSparseLocalBlocks: Int = 16,
        blockSparseVerticalStride: Int = 8
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.denseAttentionEveryN = denseAttentionEveryN
        self.intermediateSize = intermediateSize
        self.gegeluLimit = gegeluLimit
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.layerNormEps = layerNormEps
        self.vocabularySize = vocabularySize
        self.keyValueHeads = keyValueHeads
        self.mupAttentionMultiplier = mupAttentionMultiplier
        self.mupUseScaling = mupUseScaling
        self.mupEmbeddingMultiplier = mupEmbeddingMultiplier
        self.mupWidthMultiplier = mupWidthMultiplier
        self.ropeBase = ropeBase
        self.ropeScale = ropeScale
        self.blockSparseBlockSize = blockSparseBlockSize
        self.blockSparseLocalBlocks = blockSparseLocalBlocks
        self.blockSparseVerticalStride = blockSparseVerticalStride
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case denseAttentionEveryN = "dense_attention_every_n_layers"
        case intermediateSize = "ff_intermediate_size"
        case gegeluLimit = "gegelu_limit"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case layerNormEps = "layer_norm_epsilon"
        case vocabularySize = "vocab_size"
        case keyValueHeads = "num_key_value_heads"
        case mupAttentionMultiplier = "mup_attn_multiplier"
        case mupUseScaling = "mup_use_scaling"
        case mupEmbeddingMultiplier = "mup_embedding_multiplier"
        case mupWidthMultiplier = "mup_width_multiplier"
        case ropeBase = "rope_embedding_base"
        case ropeScale = "rope_position_scale"
        case blockSparseBlockSize = "blocksparse_block_size"
        case blockSparseLocalBlocks = "blocksparse_num_local_blocks"
        case blockSparseVerticalStride = "blocksparse_vert_stride"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "phi3small",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            denseAttentionEveryN: try container.decode(
                Int.self,
                forKey: .denseAttentionEveryN
            ),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            gegeluLimit: try container.decode(Float.self, forKey: .gegeluLimit),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            layerNormEps: try container.decode(Float.self, forKey: .layerNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            keyValueHeads: try container.decode(Int.self, forKey: .keyValueHeads),
            mupAttentionMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .mupAttentionMultiplier
            ) ?? 1,
            mupUseScaling: try container.decodeIfPresent(Bool.self, forKey: .mupUseScaling) ?? true,
            mupEmbeddingMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .mupEmbeddingMultiplier
            ) ?? 10,
            mupWidthMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .mupWidthMultiplier
            ) ?? 8,
            ropeBase: try container.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 1_000_000,
            ropeScale: try container.decodeIfPresent(Float.self, forKey: .ropeScale) ?? 1,
            blockSparseBlockSize: try container.decodeIfPresent(
                Int.self,
                forKey: .blockSparseBlockSize
            ) ?? 64,
            blockSparseLocalBlocks: try container.decodeIfPresent(
                Int.self,
                forKey: .blockSparseLocalBlocks
            ) ?? 16,
            blockSparseVerticalStride: try container.decodeIfPresent(
                Int.self,
                forKey: .blockSparseVerticalStride
            ) ?? 8
        )
    }
}

extension Phi3SmallModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["query_key_value"]) }
    }
}
