import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Llama 4 architecture")
struct Llama4ArchitectureTests {
    @Test("decodes tiny Llama 4 text configuration")
    func decodesTinyLlama4TextConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            Llama4TextConfiguration.self,
            from: Data(kLlama4RealTinyConfigJSON.utf8)
        )
        let layout = Llama4AttentionLayout(config)

        #expect(config.modelType == "llama4_text")
        #expect(config.hiddenSize == 8)
        #expect(config.hiddenLayers == 2)
        #expect(config.attentionHeads == 4)
        #expect(config.keyValueHeads == 2)
        #expect(config.headDimensions == 128)
        #expect(config.vocabularySize == 201_135)
        #expect(config.denseIntermediateSize == 16_384)
        #expect(config.noRopeLayers == [true, true])
        #expect(config.moeLayers == [0, 1])
        #expect(config.usesRope(at: 0))
        #expect(config.useQKNorm)
        #expect(config.tieWordEmbeddings == false)
        #expect(layout.queryProjectionSize == 512)
        #expect(layout.keyValueProjectionSize == 256)
    }

    @Test("registers Llama 4 model aliases")
    func registersLlama4ModelAliases() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(registeredTypes.contains("llama4"))
        #expect(registeredTypes.contains("llama4_text"))
    }

    @Test("constructs Llama 4 text model through the factory")
    func constructsLlama4TextModelThroughFactory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Llama4ArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try kLlama4TextFactoryConfigJSON.write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "llama4_text"
        )

        #expect(model is Llama4TextModel)
        #expect((model as? Llama4TextModel)?.vocabularySize == 64)
    }

    @Test("constructs Llama 4 wrapper model through the factory")
    func constructsLlama4WrapperModelThroughFactory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Llama4WrapperArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try kLlama4WrapperFactoryConfigJSON.write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "llama4"
        )

        #expect(model is Llama4Model)
        #expect((model as? Llama4Model)?.vocabularySize == 64)
    }

    @Test("tiny Llama 4 text model produces finite logits")
    func tinyLlama4TextModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Llama4TextModel(smallLlama4TextConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("tiny Llama 4 wrapper model produces finite logits")
    func tinyLlama4WrapperModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Llama4Model(.init(textConfig: smallLlama4WrapperConfig()))
            let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        Device.withDefaultDevice(.cpu) {
            let model = Llama4TextModel(smallLlama4TextConfig())
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            eval(output.token)

            #expect(output.token.shape == [1])
        }
    }

    @Test("sanitizer packs Llama 4 fused expert tensors")
    func sanitizerPacksLlama4FusedExpertTensors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Llama4Model(.init(textConfig: smallLlama4WrapperConfig()))
            let sanitized = model.sanitize(weights: llama4WrapperPackedExpertWeights())
            expectLlama4FusedExpertWeightsRemoved(from: sanitized)
            try expectLlama4PackedExpertWeights(in: sanitized)
        }
    }
}

private func smallLlama4TextConfig() -> Llama4TextConfiguration {
    Llama4TextConfiguration(
        hiddenSize: 8,
        attentionHeads: 4,
        hiddenLayers: 2,
        vocabularySize: 64,
        intermediateSize: 8,
        denseIntermediateSize: 16,
        keyValueHeads: 2,
        rmsNormEps: 1e-5,
        ropeTheta: 10_000,
        headDimensions: 2,
        tieWordEmbeddings: false,
        noRopeLayers: [true, false],
        useQKNorm: true,
        localExperts: 2
    )
}

private func smallLlama4WrapperConfig() -> Llama4TextConfiguration {
    Llama4TextConfiguration(
        modelType: "llama4",
        hiddenSize: 8,
        attentionHeads: 4,
        hiddenLayers: 2,
        vocabularySize: 64,
        intermediateSize: 3,
        denseIntermediateSize: 12,
        keyValueHeads: 2,
        rmsNormEps: 1e-5,
        ropeTheta: 10_000,
        headDimensions: 2,
        tieWordEmbeddings: false,
        noRopeLayers: [true, false],
        useQKNorm: true,
        localExperts: 2
    )
}

private func llama4WrapperPackedExpertWeights() -> [String: MLXArray] {
    [
        kLlama4FusedGateUpKey: MLXArray.ones([2, 8, 6]),
        kLlama4FusedDownKey: MLXArray.ones([2, 3, 8])
    ]
}

private func expectLlama4FusedExpertWeightsRemoved(from weights: [String: MLXArray]) {
    #expect(weights[kLlama4FusedGateUpKey] == nil)
    #expect(weights[kLlama4FusedDownKey] == nil)
}

private func expectLlama4PackedExpertWeights(in weights: [String: MLXArray]) throws {
    let gate = try #require(weights[kLlama4GateWeightKey])
    let upProjection = try #require(weights[kLlama4UpWeightKey])
    let down = try #require(weights[kLlama4DownWeightKey])

    eval(gate, upProjection, down)
    #expect(gate.shape == [2, 3, 8])
    #expect(upProjection.shape == [2, 3, 8])
    #expect(down.shape == [2, 8, 3])
}

private let kLlama4FusedGateUpKey = "language_model.model.layers.0.feed_forward.experts.gate_up_proj"
private let kLlama4FusedDownKey = "language_model.model.layers.0.feed_forward.experts.down_proj"
private let kLlama4GateWeightKey = "language_model.model.layers.0.feed_forward.experts.gate_proj.weight"
private let kLlama4UpWeightKey = "language_model.model.layers.0.feed_forward.experts.up_proj.weight"
private let kLlama4DownWeightKey = "language_model.model.layers.0.feed_forward.experts.down_proj.weight"

private let kLlama4TextFactoryConfigJSON = #"""
    {
        "model_type": "llama4_text",
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "intermediate_size": 8,
        "intermediate_size_mlp": 16,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "head_dim": 2,
        "tie_word_embeddings": false,
        "no_rope_layers": [1, 0],
        "use_qk_norm": true,
        "num_local_experts": 2
    }
    """#

private let kLlama4WrapperFactoryConfigJSON = #"""
    {
        "model_type": "llama4",
        "text_config": {
            "model_type": "llama4",
            "hidden_size": 8,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "vocab_size": 64,
            "intermediate_size": 3,
            "intermediate_size_mlp": 12,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "head_dim": 2,
            "tie_word_embeddings": false,
            "no_rope_layers": [1, 0],
            "use_qk_norm": true,
            "num_local_experts": 2
        }
    }
    """#

private let kLlama4RealTinyConfigJSON = #"""
    {
        "architectures": [
            "Llama4ForCausalLM"
        ],
        "attention_bias": false,
        "attention_chunk_size": 8192,
        "attn_scale": 0.1,
        "attn_temperature_tuning": true,
        "floor_scale": 8192,
        "head_dim": 128,
        "hidden_size": 8,
        "interleave_moe_layer_step": 1,
        "intermediate_size": 32,
        "intermediate_size_mlp": 16384,
        "max_position_embeddings": 131072,
        "model_type": "llama4_text",
        "moe_layers": [
            0,
            1
        ],
        "no_rope_layers": [
            1,
            1
        ],
        "num_attention_heads": 4,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 16,
        "rms_norm_eps": 0.00001,
        "rope_scaling": null,
        "rope_theta": 500000,
        "tie_word_embeddings": false,
        "use_qk_norm": true,
        "vocab_size": 201135
    }
    """#
