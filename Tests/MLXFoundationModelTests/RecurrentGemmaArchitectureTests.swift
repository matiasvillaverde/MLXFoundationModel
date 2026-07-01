import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("RecurrentGemma architecture")
struct RecurrentGemmaArchitectureTests {
    @Test("decodes configuration and fallback block types")
    func decodesConfigurationAndFallbackBlockTypes() throws {
        let config = try JSONDecoder.json5().decode(
            RecurrentGemmaConfiguration.self,
            from: Data(Self.configJSON(useUnderscoredBlockTypes: true).utf8)
        )

        #expect(config.modelType == "recurrent_gemma")
        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 3)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.headDim == 4)
        #expect(config.convolutionWidth == 3)
        #expect(config.attentionWindowSize == 8)
        #expect(config.blockTypes == ["recurrent", "attention"])
        #expect(config.tieWordEmbeddings)
    }

    @Test("builds layer and cache plan")
    func buildsLayerAndCachePlan() {
        let plan = RecurrentGemmaLayerPlan(Self.smallConfig(hiddenLayers: 4))

        #expect(plan.blockTypes == ["recurrent", "attention", "recurrent", "attention"])
        #expect(plan.firstAttentionIndex == 1)
        #expect(plan.cacheKinds == ["recurrent", "attention", "recurrent", "attention"])
    }

    @Test("registers and constructs RecurrentGemma through the factory")
    func registersAndConstructsRecurrentGemmaThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("recurrent_gemma"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RecurrentGemmaArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.configJSON(useUnderscoredBlockTypes: false).write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "recurrent_gemma"
        )

        #expect(model is RecurrentGemmaModel)
        #expect((model as? RecurrentGemmaModel)?.vocabularySize == 64)
    }

    @Test("constructs mixed caches, adapters, and greedy fast path")
    func constructsMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = RecurrentGemmaModel(Self.smallConfig(hiddenLayers: 3))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "recurrent_gemma")
        #expect(model.vocabularySize == 64)
        #expect(cache.count == 3)
        _ = try #require(cache[0] as? MambaCache)
        _ = try #require(cache[1] as? RotatingKVCache)
        _ = try #require(cache[2] as? MambaCache)
        #expect(loraTargets.count == 6)
        #expect(loraTargets[0].1 == ["linear_x", "linear_y", "linear_out"])
        #expect(loraTargets[2].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny tied model produces finite logits with and without cache")
    func tinyTiedModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = RecurrentGemmaModel(Self.smallConfig())
            let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
            let logits = model(tokens, cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))

            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache.map(\.offset) == [3, 3, 3])
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("tiny untied model uses language head")
    func tinyUntiedModelUsesLanguageHead() {
        Device.withDefaultDevice(.cpu) {
            let model = RecurrentGemmaModel(Self.smallConfig(tieWordEmbeddings: false))
            let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer reshapes convolution weights and strips tied head")
    func sanitizerReshapesConvolutionWeightsAndStripsTiedHead() {
        let model = RecurrentGemmaModel(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.temporal_block.conv_1d.weight": MLXArray.ones([16, 1, 3]),
            "model.layers.0.temporal_block.rotary_emb.inv_freq": MLXArray.ones([1]),
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.layers.0.temporal_block.conv_1d.weight"]?.shape == [16, 3, 1])
        #expect(sanitized["model.layers.0.temporal_block.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 3,
        tieWordEmbeddings: Bool = true
    ) -> RecurrentGemmaConfiguration {
        RecurrentGemmaConfiguration(
            convolutionWidth: 3,
            hiddenSize: 16,
            intermediateSize: 32,
            logitsSoftCap: 30,
            attentionHeads: 4,
            hiddenLayers: hiddenLayers,
            keyValueHeads: 1,
            attentionWindowSize: 8,
            vocabularySize: 64,
            blockTypes: ["recurrent", "attention"],
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func configJSON(useUnderscoredBlockTypes: Bool) -> String {
        let blockKey = useUnderscoredBlockTypes ? "_block_types" : "block_types"
        return """
        {
            "model_type": "recurrent_gemma",
            "attention_bias": true,
            "conv1d_width": 3,
            "hidden_size": 16,
            "intermediate_size": 32,
            "logits_soft_cap": 30,
            "num_attention_heads": 4,
            "num_hidden_layers": 3,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
            "attention_window_size": 8,
            "vocab_size": 64,
            "embeddings_scale_by_sqrt_dim": true,
            "tie_word_embeddings": true,
            "\(blockKey)": ["recurrent", "attention"]
        }
        """
    }
}
