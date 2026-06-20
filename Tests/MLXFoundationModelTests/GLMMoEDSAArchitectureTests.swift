import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("GLM MoE DSA architecture")
struct GLMMoEDSAArchitectureTests {
    @Test("decodes explicit full shared indexer schedule")
    func decodesExplicitFullSharedIndexerSchedule() throws {
        let config = try Self.decodeConfig(indexSchedule: """
        "index_topk_pattern": "FSFS"
        """)

        #expect(config.modelType == "glm_moe_dsa")
        #expect(config.usesDSA)
        #expect(config.dsaIndexerKinds == [.full, .shared, .full, .shared])
        #expect(config.ropeTheta == 1_000_000)
        #expect(config.ropeScaling?["rope_theta"]?.asFloat() == 1_000_000)
    }

    @Test("derives indexer schedule from frequency and skip offset")
    func derivesIndexerScheduleFromFrequencyAndSkipOffset() throws {
        let config = try Self.decodeConfig(indexSchedule: """
        "index_topk_freq": 3,
        "index_skip_topk_offset": 2
        """)

        #expect(config.dsaIndexerKinds == [.full, .full, .shared, .shared])
    }

    @Test("rejects invalid indexer schedule")
    func rejectsInvalidIndexerSchedule() {
        #expect(throws: DecodingError.self) {
            _ = try Self.decodeConfig(indexSchedule: """
            "index_topk_pattern": "SFSF"
            """)
        }
    }

    @Test("registers GLM DSA model type")
    func registersGLMDSAModelType() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(registeredTypes.contains("glm_moe_dsa"))
    }

    @Test(
        "constructs GLM DSA model with full and shared cache layout",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func constructsGLMDSAModelWithCacheLayout() throws {
        let directory = try Self.writeModelDirectory(indexSchedule: """
        "indexer_types": ["full", "shared", "full", "shared"]
        """)
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try #require(
            LLMTypeRegistry.shared.createModel(
                configuration: directory.appendingPathComponent("config.json"),
                modelType: "glm_moe_dsa"
            ) as? GLM4MoELiteModel
        )
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 4)
        #expect((cache[0] as? CacheList)?.layoutCaches.count == 2)
        #expect((cache[1] as? CacheList)?.layoutCaches.count == 1)
        #expect((cache[2] as? CacheList)?.layoutCaches.count == 2)
        #expect((cache[3] as? CacheList)?.layoutCaches.count == 1)
    }

    @Test(
        "tiny GLM DSA model produces finite logits through full and shared indexers",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func tinyGLMDSAModelProducesFiniteLogits() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try Self.decodeConfig(indexSchedule: """
            "index_topk_pattern": "FSFS"
            """)
            let model = GLM4MoELiteModel(config)
            let cache = model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)]).reshaped([1, 4])

            let logits = model(prompt, cache: cache)
            eval(logits)

            #expect(logits.shape == [1, 4, config.vocabularySize])
            #expect(all(isFinite(logits)).item(Bool.self))
            #expect((cache[0] as? CacheList)?.layoutCaches[0].offset == 4)
            #expect((cache[0] as? CacheList)?.layoutCaches[1].offset == 4)
            #expect((cache[1] as? CacheList)?.layoutCaches[0].offset == 4)
        }
    }

    private static func decodeConfig(indexSchedule: String) throws -> GLM4MoELiteConfiguration {
        try JSONDecoder.json5().decode(
            GLM4MoELiteConfiguration.self,
            from: Self.configJSON(indexSchedule: indexSchedule)
        )
    }

    private static func writeModelDirectory(indexSchedule: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("GLMMoEDSAArchitectureTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.configJSON(indexSchedule: indexSchedule)
            .write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private static func configJSON(indexSchedule: String) -> Data {
        Data("\(Self.baseConfigPrefix)\(indexSchedule)\n\(Self.baseConfigSuffix)".utf8)
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
            "model_type": "glm_moe_dsa",
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
            "q_lora_rank": 6,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "rms_norm_eps": 0.00001,
            "rope_parameters": {"rope_theta": 1000000.0},
            "routed_scaling_factor": 1.0,
            "scoring_func": "sigmoid",
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "v_head_dim": 4,
            "vocab_size": 64,
        """

    private static let baseConfigSuffix = """
        }
        """
}
