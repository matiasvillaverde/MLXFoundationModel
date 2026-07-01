import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Nemotron-NAS architecture")
struct NemotronNASArchitectureTests {
    @Test("decodes heterogeneous block configuration")
    func decodesHeterogeneousBlockConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronNASConfiguration.self,
            from: Data(kNemotronNASConfigJSON.utf8)
        )

        #expect(config.modelType == "nemotron-nas")
        #expect(config.hiddenSize == 8_192)
        #expect(config.hiddenLayers == 4)
        #expect(config.attentionHeads == 64)
        #expect(config.vocabularySize == 128_256)
        #expect(config.blockConfigs[0].attention.headsPerKVGroup == 8)
        #expect(config.blockConfigs[1].attention.replaceWithLinear)
        #expect(config.blockConfigs[2].attention.noOp)
        #expect(config.blockConfigs[3].feedForward.noOp)
        #expect(config.realAttentionLayerCount == 2)
    }

    @Test("builds grouped attention layout")
    func buildsGroupedAttentionLayout() {
        let config = Self.smallConfig()
        let layout = NemotronNASAttentionLayout(
            config,
            attention: config.blockConfigs[0].attention
        )

        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDim == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("registers and constructs Nemotron-NAS through the factory")
    func registersAndConstructsNemotronNASThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("nemotron-nas"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NemotronNASArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try kNemotronNASTinyConfigJSON.write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "nemotron-nas"
        )

        #expect(model is NemotronNASModel)
        #expect((model as? NemotronNASModel)?.vocabularySize == 64)
    }

    @Test("constructs cache, adapters, and greedy fast path")
    func constructsCacheAdaptersAndGreedyFastPath() {
        let model = NemotronNASModel(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "nemotron-nas")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 4])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny model produces finite logits with cache")
    func tinyModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = NemotronNASModel(Self.smallConfig())
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(cache[1].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("tiny tied model uses embedding head")
    func tinyTiedModelUsesEmbeddingHead() {
        Device.withDefaultDevice(.cpu) {
            let model = NemotronNASModel(Self.smallConfig(tieWordEmbeddings: true))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer strips rotary metadata and tied output head")
    func sanitizerStripsRotaryMetadataAndTiedOutputHead() {
        let model = NemotronNASModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([1]),
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    private static func smallConfig(
        tieWordEmbeddings: Bool = false
    ) -> NemotronNASConfiguration {
        NemotronNASConfiguration(
            hiddenSize: 16,
            hiddenLayers: 4,
            attentionHeads: 4,
            vocabularySize: 64,
            blockConfigs: [
                .init(
                    attention: .init(headsPerKVGroup: 2),
                    feedForward: .init(multiplier: 1)
                ),
                .init(
                    attention: .init(replaceWithLinear: true),
                    feedForward: .init(replaceWithLinear: true)
                ),
                .init(
                    attention: .init(noOp: true),
                    feedForward: .init(multiplier: 1)
                ),
                .init(
                    attention: .init(headsPerKVGroup: 1),
                    feedForward: .init(noOp: true)
                )
            ],
            maxPositionEmbeddings: 64,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}

private let kNemotronNASTinyConfigJSON = #"""
{
    "model_type": "nemotron-nas",
    "hidden_size": 16,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": 64,
    "hidden_act": "silu",
    "attention_bias": false,
    "mlp_bias": false,
    "rope_theta": 10000,
    "max_position_embeddings": 64,
    "tie_word_embeddings": false,
    "block_configs": [
        {
            "attention": {"n_heads_in_group": 2},
            "ffn": {"ffn_mult": 1.0}
        },
        {
            "attention": {"replace_with_linear": true},
            "ffn": {"replace_with_linear": true}
        },
        {
            "attention": {"no_op": true},
            "ffn": {"ffn_mult": 1.0}
        },
        {
            "attention": {"n_heads_in_group": 1},
            "ffn": {"no_op": true}
        }
    ]
}
"""#

private let kNemotronNASConfigJSON = #"""
{
    "model_type": "nemotron-nas",
    "hidden_size": 8192,
    "num_hidden_layers": 4,
    "num_attention_heads": 64,
    "rms_norm_eps": 1e-5,
    "vocab_size": 128256,
    "hidden_act": "silu",
    "attention_bias": false,
    "mlp_bias": false,
    "rope_theta": 500000,
    "max_position_embeddings": 131072,
    "tie_word_embeddings": false,
    "block_configs": [
        {
            "attention": {"n_heads_in_group": 8},
            "ffn": {"ffn_mult": 1.25}
        },
        {
            "attention": {"replace_with_linear": true},
            "ffn": {"replace_with_linear": true}
        },
        {
            "attention": {"no_op": true},
            "ffn": {"ffn_mult": 1.0}
        },
        {
            "attention": {"n_heads_in_group": 4},
            "ffn": {"no_op": true}
        }
    ]
}
"""#
