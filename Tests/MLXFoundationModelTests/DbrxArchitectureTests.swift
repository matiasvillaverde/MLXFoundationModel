import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("DBRX architecture")
struct DbrxArchitectureTests {
    @Test("decodes DBRX configuration")
    func decodesDBRXConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            DbrxConfiguration.self,
            from: Data(Self.realTinyConfigJSON.utf8)
        )
        let layout = DbrxAttentionLayout(config)

        #expect(config.modelType == "dbrx")
        #expect(config.vocabularySize == 100_352)
        #expect(config.hiddenSize == 4)
        #expect(config.hiddenLayers == 2)
        #expect(config.attentionHeads == 4)
        #expect(config.attentionConfig.keyValueHeads == 2)
        #expect(config.attentionConfig.clipQKV == 8)
        #expect(config.attentionConfig.ropeTheta == 500_000)
        #expect(config.feedForwardConfig.hiddenSize == 8)
        #expect(config.feedForwardConfig.expertCount == 16)
        #expect(config.feedForwardConfig.expertsPerToken == 4)
        #expect(config.layerNormEpsilon == 1e-5)
        #expect(config.tieWordEmbeddings == false)
        #expect(layout.headSize == 1)
        #expect(layout.packedProjectionSize == 8)
    }

    @Test("builds DBRX attention layout and routing plan")
    func buildsDBRXAttentionLayoutAndRoutingPlan() {
        let config = Self.smallConfig()
        let layout = DbrxAttentionLayout(config)
        let routing = DbrxRoutingPlan(config)

        #expect(layout.headSize == 2)
        #expect(layout.queryProjectionSize == 8)
        #expect(layout.keyValueProjectionSize == 4)
        #expect(layout.packedProjectionSize == 16)
        #expect(layout.attentionScale == 0.70710677)
        #expect(routing.expertCount == 4)
        #expect(routing.expertsPerToken == 2)
    }

    @Test("routes top-k experts with normalized weights")
    func routesTopKExpertsWithNormalizedWeights() {
        Device.withDefaultDevice(.cpu) {
            let routing = DbrxRoutingPlan(Self.smallConfig())
            let logits = MLXArray([Float(0), 5, 1, 4]).reshaped(1, 1, 4)
            let routed = routing.route(logits)
            eval(routed.indices, routed.scores)

            let selected = Set(routed.indices.asArray(Int32.self).map(Int.init))
            let scores = routed.scores.asArray(Float.self)
            #expect(selected == [1, 3])
            #expect(abs(scores.reduce(0, +) - 1) < 0.0001)
        }
    }

    @Test("constructs DBRX model through the factory")
    func constructsDBRXModelThroughFactory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("DbrxArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.factoryConfigJSON.write(to: configurationURL, atomically: true, encoding: .utf8)

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "dbrx"
        )

        #expect(model is DbrxModel)
        #expect((model as? DbrxModel)?.vocabularySize == 64)
    }

    @Test("tiny DBRX model produces finite logits")
    func tinyDBRXModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = DbrxModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        Device.withDefaultDevice(.cpu) {
            let model = DbrxModel(Self.smallConfig())
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            eval(output.token)

            #expect(output.token.shape == [1])
        }
    }

    @Test("sanitizer packs DBRX fused expert tensors")
    func sanitizerPacksDBRXFusedExpertTensors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = DbrxModel(Self.smallConfig(expertCount: 2, feedForwardHiddenSize: 3))
            let sanitized = model.sanitize(weights: Self.packedExpertWeights())

            #expect(sanitized["transformer.blocks.0.ffn.experts.mlp.w1"] == nil)
            #expect(sanitized["transformer.blocks.0.ffn.experts.mlp.v1"] == nil)
            #expect(sanitized["transformer.blocks.0.ffn.experts.mlp.w2"] == nil)

            let gate = try #require(
                sanitized["transformer.blocks.0.ffn.switch_mlp.gate_proj.weight"]
            )
            let upProjection = try #require(
                sanitized["transformer.blocks.0.ffn.switch_mlp.up_proj.weight"]
            )
            let down = try #require(
                sanitized["transformer.blocks.0.ffn.switch_mlp.down_proj.weight"]
            )

            eval(gate, upProjection, down)
            #expect(gate.shape == [2, 3, 8])
            #expect(upProjection.shape == [2, 3, 8])
            #expect(down.shape == [2, 8, 3])
        }
    }

    private static func smallConfig(
        expertCount: Int = 4,
        feedForwardHiddenSize: Int = 16
    ) -> DbrxConfiguration {
        DbrxConfiguration(
            vocabularySize: 64,
            hiddenSize: 8,
            hiddenLayers: 2,
            attentionHeads: 4,
            attentionConfig: .init(keyValueHeads: 2, clipQKV: 8, ropeTheta: 10_000),
            feedForwardConfig: .init(
                hiddenSize: feedForwardHiddenSize,
                expertCount: expertCount,
                expertsPerToken: 2
            )
        )
    }

    private static func packedExpertWeights() -> [String: MLXArray] {
        [
            "transformer.blocks.0.ffn.experts.mlp.w1": MLXArray.ones([6, 8]),
            "transformer.blocks.0.ffn.experts.mlp.v1": MLXArray.ones([6, 8]),
            "transformer.blocks.0.ffn.experts.mlp.w2": MLXArray.ones([6, 8])
        ]
    }

    private static let factoryConfigJSON = #"""
    {
        "model_type": "dbrx",
        "vocab_size": 64,
        "d_model": 8,
        "n_layers": 2,
        "n_heads": 4,
        "attn_config": {
            "clip_qkv": 8,
            "kv_n_heads": 2,
            "rope_theta": 10000
        },
        "ffn_config": {
            "ffn_hidden_size": 16,
            "moe_num_experts": 4,
            "moe_top_k": 2
        },
        "tie_word_embeddings": false
    }
    """#

    private static let realTinyConfigJSON = #"""
    {
        "attn_config": {
            "clip_qkv": 8,
            "kv_n_heads": 2,
            "rope_theta": 500000
        },
        "d_model": 4,
        "ffn_config": {
            "ffn_hidden_size": 8,
            "moe_num_experts": 16,
            "moe_top_k": 4
        },
        "model_type": "dbrx",
        "n_heads": 4,
        "n_layers": 2,
        "tie_word_embeddings": false,
        "vocab_size": 100352
    }
    """#
}
