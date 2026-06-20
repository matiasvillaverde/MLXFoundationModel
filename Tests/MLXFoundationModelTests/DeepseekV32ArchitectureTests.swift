import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("DeepSeek V3.2 architecture")
struct DeepseekV32ArchitectureTests {
    @Test("decodes DeepSeek V3.2 as all-full DSA schedule")
    func decodesDeepseekV32AsAllFullDSASchedule() throws {
        let config = try Self.decodeConfig()

        #expect(config.modelType == "deepseek_v32")
        #expect(config.usesDSA)
        #expect(config.dsaIndexerKinds == [.full, .full, .full, .full])
    }

    @Test("registers DeepSeek V3.2 model type")
    func registersDeepseekV32ModelType() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(registeredTypes.contains("deepseek_v32"))
    }

    @Test(
        "constructs DeepSeek V3.2 model with DSA cache layout",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func constructsDeepseekV32ModelWithCacheLayout() throws {
        let directory = try Self.writeModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try #require(
            LLMTypeRegistry.shared.createModel(
                configuration: directory.appendingPathComponent("config.json"),
                modelType: "deepseek_v32"
            ) as? DeepseekV32Model
        )
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 4)
        #expect(cache.allSatisfy { ($0 as? CacheList)?.layoutCaches.count == 2 })
    }

    @Test(
        "constructs DeepSeek V3.2 model with IndexCache shared layers",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func constructsDeepseekV32ModelWithIndexCacheLayout() throws {
        let directory = try Self.writeModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try #require(
            LLMTypeRegistry.shared.createModel(
                configuration: directory.appendingPathComponent("config.json"),
                modelType: "deepseek_v32"
            ) as? DeepseekV32Model
        )
        let cache = model.newCache(parameters: GenerateParameters(indexCacheFrequency: 2))

        #expect(cache.count == 4)
        #expect((cache[0] as? CacheList)?.layoutCaches.count == 2)
        #expect((cache[1] as? CacheList)?.layoutCaches.count == 1)
        #expect((cache[2] as? CacheList)?.layoutCaches.count == 2)
        #expect((cache[3] as? CacheList)?.layoutCaches.count == 1)
    }

    @Test(
        "tiny DeepSeek V3.2 model produces finite logits through DSA indexers",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func tinyDeepseekV32ModelProducesFiniteLogits() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try Self.decodeConfig()
            let model = DeepseekV32Model(config)
            let cache = model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)]).reshaped([1, 4])

            let logits = model(prompt, cache: cache)
            eval(logits)

            #expect(logits.shape == [1, 4, config.vocabularySize])
            #expect(all(isFinite(logits)).item(Bool.self))
            #expect((cache[0] as? CacheList)?.layoutCaches[0].offset == 4)
            #expect((cache[0] as? CacheList)?.layoutCaches[1].offset == 4)
            #expect((cache[1] as? CacheList)?.layoutCaches[0].offset == 4)
            #expect((cache[1] as? CacheList)?.layoutCaches[1].offset == 4)
        }
    }

    @Test(
        "tiny DeepSeek V3.2 model decodes with IndexCache shared layers",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func tinyDeepseekV32ModelDecodesWithIndexCacheSharedLayers() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try Self.decodeConfig()
            let model = DeepseekV32Model(config)
            let cache = model.newCache(parameters: GenerateParameters(indexCacheFrequency: 2))
            let prompt = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)]).reshaped([1, 4])

            let promptLogits = model(prompt, cache: cache)
            let decodeLogits = model(MLXArray([Int32(5)]).reshaped([1, 1]), cache: cache)
            eval(promptLogits, decodeLogits)

            #expect(promptLogits.shape == [1, 4, config.vocabularySize])
            #expect(decodeLogits.shape == [1, 1, config.vocabularySize])
            #expect(all(isFinite(decodeLogits)).item(Bool.self))
            #expect((cache[0] as? CacheList)?.layoutCaches[1].offset == 5)
            #expect((cache[1] as? CacheList)?.layoutCaches.count == 1)
            #expect((cache[1] as? CacheList)?.layoutCaches[0].offset == 5)
        }
    }

    @Test(
        "sanitizer converts DeepSeek V3.2 kv_b_proj and strips MTP layers",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func sanitizerConvertsKVProjectionAndStripsMTPLayers() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try Self.decodeConfig(numNextPredictLayers: 1)
            let model = DeepseekV32Model(config)
            let kvBWeight = MLXArray.ones([16, 4], type: Float32.self)
            let kvBScaleInv = MLXArray(Array(repeating: Float(2), count: 64)).reshaped(16, 4)
            let mtpWeight = MLXArray.zeros([config.hiddenSize, config.hiddenSize])

            let sanitized = model.sanitize(weights: [
                "model.layers.0.self_attn.kv_b_proj.weight": kvBWeight,
                "model.layers.0.self_attn.kv_b_proj.weight_scale_inv": kvBScaleInv,
                "model.layers.4.self_attn.q_proj.weight": mtpWeight
            ])

            #expect(sanitized["model.layers.0.self_attn.kv_b_proj.weight"] == nil)
            #expect(sanitized["model.layers.0.self_attn.kv_b_proj.weight_scale_inv"] == nil)
            let embedQ = try #require(sanitized["model.layers.0.self_attn.embed_q.weight"])
            eval(embedQ)
            #expect(embedQ.shape == [2, 4, 4])
            #expect(embedQ.asArray(Float.self).allSatisfy { $0 == 2 })
            #expect(sanitized["model.layers.0.self_attn.unembed_out.weight"]?.shape == [2, 4, 4])
            #expect(sanitized["model.layers.4.self_attn.q_proj.weight"] == nil)
        }
    }

    private static func decodeConfig(numNextPredictLayers: Int = 0) throws -> DeepseekV32Configuration {
        try JSONDecoder.json5().decode(
            DeepseekV32Configuration.self,
            from: Self.configJSON(numNextPredictLayers: numNextPredictLayers)
        )
    }

    private static func writeModelDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("DeepseekV32ArchitectureTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.configJSON().write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private static func configJSON(numNextPredictLayers: Int = 0) -> Data {
        Data("\(Self.baseConfigPrefix)\(numNextPredictLayers)\(Self.baseConfigSuffix)".utf8)
    }

    private static let baseConfigPrefix = """
        {
            "attention_bias": false,
            "first_k_dense_replace": 1,
            "hidden_size": 16,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_topk": 2,
            "intermediate_size": 32,
            "kv_lora_rank": 4,
            "max_position_embeddings": 64,
            "model_type": "deepseek_v32",
            "moe_intermediate_size": 8,
            "moe_layer_freq": 1,
            "n_group": 1,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "norm_topk_prob": true,
            "num_attention_heads": 2,
            "num_experts_per_tok": 1,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "num_nextn_predict_layers":
        """

    private static let baseConfigSuffix = """
        ,
            "q_lora_rank": 6,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "rms_norm_eps": 0.00001,
            "rope_theta": 1000000.0,
            "routed_scaling_factor": 1.0,
            "scoring_func": "sigmoid",
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "v_head_dim": 4,
            "vocab_size": 64
        }
        """
}
