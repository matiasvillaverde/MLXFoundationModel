import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("AFM7 architecture")
struct AFM7ArchitectureTests {
    @Test("decodes AFM7 configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            AFM7Configuration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "afm7")
        #expect(config.vocabularySize == 64)
        #expect(config.hiddenDim == 16)
        #expect(config.layerCount == 3)
        #expect(config.kvReuseLayerCount == 1)
        #expect(config.baseLayerCount == 2)
        #expect(config.hiddenDimScaleFactor == 3.25)
        #expect(config.feedForwardSize == 52)
        #expect(config.ropeTheta == 50_000)
        #expect(config.rmsNormEps == 1e-5)
    }

    @Test("builds grouped attention layout")
    func buildsGroupedAttentionLayout() {
        let layout = AFM7AttentionLayout(Self.smallConfig())

        #expect(layout.hiddenDim == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDim == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.combinedProjectionSize == 32)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("registers and constructs AFM7 through the factory")
    func registersAndConstructsAFM7ThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("afm7"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFM7ArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.configJSON(includeOptionalFields: true).write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "afm7"
        )

        #expect(model is AFM7Model)
        #expect((model as? AFM7Model)?.vocabularySize == 64)
    }

    @Test("constructs cache, adapters, and greedy fast path")
    func constructsCacheAdaptersAndGreedyFastPath() {
        let model = AFM7Model(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "afm7")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 3)
        #expect(loraTargets[0].1 == ["qkv_proj", "out_proj"])
        #expect(loraTargets[2].1 == ["q_proj", "out_proj"])
    }

    @Test("tiny model produces finite logits with and without cache")
    func tinyModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = AFM7Model(Self.smallConfig())
            let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
            let logits = model(tokens, cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))

            let cache = model.newCache(parameters: nil)
            let cachedPrefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let cachedNext = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(cachedPrefill, cachedNext)

            #expect(cachedPrefill.shape == [1, 2, 64])
            #expect(cachedNext.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(cache[1].offset == 3)
            #expect(all(isFinite(cachedNext)).item(Bool.self))
        }
    }

    @Test("sanitizer strips rotary metadata and output heads")
    func sanitizerStripsRotaryMetadataAndOutputHeads() {
        let model = AFM7Model(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([1]),
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.embedding.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
        #expect(sanitized["model.embedding.weight"] != nil)
    }

    private static func smallConfig() -> AFM7Configuration {
        AFM7Configuration(
            vocabularySize: 64,
            hiddenDim: 16,
            layerCount: 3,
            kvReuseLayerCount: 1,
            attentionHeads: 4,
            kvHeads: 2,
            ropeTheta: 10_000
        )
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "afm7",
            "vocab_size": 64,
            "hidden_dim": 16,
            "num_layers": 3,
            "num_kv_reuse_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 2
            \(includeOptionalFields ? ",\n\"hidden_dim_scale_factor\": 2.0" : "")
            \(includeOptionalFields ? ",\n\"rope_theta\": 10000" : "")
            \(includeOptionalFields ? ",\n\"rms_norm_eps\": 1e-6" : "")
        }
        """
    }
}
