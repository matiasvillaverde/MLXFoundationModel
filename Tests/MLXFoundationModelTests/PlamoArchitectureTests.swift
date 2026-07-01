import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Plamo architecture")
struct PlamoArchitectureTests {
    @Test("decodes legacy Plamo 13B configuration")
    func decodesLegacyPlamoConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            PlamoConfiguration.self,
            from: Data(Self.legacyConfigJSON.utf8)
        )
        let layout = PlamoAttentionLayout(config)

        #expect(config.modelType == "plamo")
        #expect(config.hiddenSize == 5_120)
        #expect(config.hiddenLayers == 40)
        #expect(config.intermediateSize == 16_640)
        #expect(config.attentionHeads == 40)
        #expect(config.sharedHeadGroupSize == 8)
        #expect(config.keyValueHeads == 5)
        #expect(config.vocabularySize == 50_432)
        #expect(!config.tieWordEmbeddings)
        #expect(layout.queryProjectionSize == 5_120)
        #expect(layout.keyValueProjectionSize == 640)
        #expect(layout.attentionScale == 0.088388346)
    }

    @Test("builds shared-head attention layout")
    func buildsSharedHeadAttentionLayout() {
        let layout = PlamoAttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.sharedHeadGroupSize == 2)
        #expect(layout.headDimensions == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("registers and constructs Plamo through the factory")
    func registersAndConstructsPlamoThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("plamo"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PlamoArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.factoryConfigJSON.write(to: configurationURL, atomically: true, encoding: .utf8)

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "plamo"
        )

        #expect(model is PlamoModel)
        #expect((model as? PlamoModel)?.vocabularySize == 64)
    }

    @Test("constructs cache, adapters, and greedy fast path")
    func constructsCacheAdaptersAndGreedyFastPath() {
        let model = PlamoModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny untied model produces finite logits with and without cache")
    func tinyUntiedModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = PlamoModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))

            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        Device.withDefaultDevice(.cpu) {
            let model = PlamoModel(Self.smallConfig())
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            eval(output.token)

            #expect(output.token.shape == [1])
        }
    }

    @Test("sanitizer strips unused rotary and tied output head tensors")
    func sanitizerStripsUnusedRotaryAndTiedHeadTensors() {
        Device.withDefaultDevice(.cpu) {
            let model = PlamoModel(Self.smallConfig(tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: [
                "lm_head.weight": MLXArray.ones([64, 16]),
                "lm_head.scales": MLXArray.ones([64, 1]),
                "lm_head.biases": MLXArray.zeros([64, 1]),
                "model.layers.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
                "model.embed_tokens.weight": MLXArray.ones([64, 16])
            ])

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["lm_head.scales"] == nil)
            #expect(sanitized["lm_head.biases"] == nil)
            #expect(sanitized["model.layers.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.embed_tokens.weight"] != nil)
        }
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        tieWordEmbeddings: Bool = false
    ) -> PlamoConfiguration {
        PlamoConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            vocabularySize: 64,
            sharedHeadGroupSize: 2,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static let factoryConfigJSON = """
    {
        "model_type": "plamo",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "n_shared_head": 2,
        "rms_norm_eps": 0.000001,
        "tie_word_embeddings": false,
        "vocab_size": 64
    }
    """

    private static let legacyConfigJSON = """
    {
        "architectures": ["PlamoForCausalLM"],
        "hidden_size": 5120,
        "intermediate_size": 16640,
        "max_position_embeddings": 4096,
        "model_type": "plamo",
        "n_shared_head": 8,
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 40,
        "rms_norm_eps": 0.000001,
        "tie_word_embeddings": false,
        "tokenizer_class": "PlamoTokenizer",
        "torch_dtype": "bfloat16",
        "vocab_size": 50432
    }
    """
}
