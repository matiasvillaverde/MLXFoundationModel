import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("GLM MoE DSA architecture")
struct GLMMoEDSAArchitectureTests {
    @Test("decodes explicit full shared indexer schedule")
    func decodesExplicitFullSharedIndexerSchedule() throws {
        let config = try GLMMoEDSATestSupport.decodeConfig(indexSchedule: """
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
        let config = try GLMMoEDSATestSupport.decodeConfig(indexSchedule: """
        "index_topk_freq": 3,
        "index_skip_topk_offset": 2
        """)

        #expect(config.dsaIndexerKinds == [.full, .full, .shared, .shared])
    }

    @Test("rejects invalid indexer schedule")
    func rejectsInvalidIndexerSchedule() {
        #expect(throws: DecodingError.self) {
            _ = try GLMMoEDSATestSupport.decodeConfig(indexSchedule: """
            "index_topk_pattern": "SFSF"
            """)
        }
    }

    @Test("registers GLM DSA model type")
    func registersGLMDSAModelType() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(registeredTypes.contains("glm_moe_dsa"))
    }

    @Test("builds GLM MoE Lite attention, layer, DSA, and routing plans")
    func buildsAttentionLayerDSAAndRoutingPlans() throws {
        let config = try GLMMoEDSATestSupport.decodeConfig(indexSchedule: """
        "index_topk_pattern": "FSFS"
        """)
        let attention = GLM4MoELiteAttentionLayout(config)
        let layers = GLM4MoELiteLayerPlan(config)
        let dsa = GLM4MoELiteDSAPlan(config)
        let routing = GLM4MoELiteRoutingPlan(config)

        #expect(attention.queryHeadDim == 8)
        #expect(attention.queryProjectionDimensions == 16)
        #expect(attention.compressedKeyValueDimensions == 8)
        #expect(attention.outputProjectionDimensions == 8)
        #expect(abs(attention.attentionScale - 0.35355338) < 0.0001)
        #expect(layers.usesSparseExperts(layerIndex: 0) == false)
        #expect(layers.usesSparseExperts(layerIndex: 1))
        #expect(dsa.kind(for: 0) == .full)
        #expect(dsa.kind(for: 1) == .shared)
        #expect(routing.expertCount == 2)
        #expect(routing.expertsPerGroup == 2)
    }

    @Test("router uses correction bias for selection only")
    func routerUsesCorrectionBiasForSelectionOnly() throws {
        let config = try GLMMoEDSATestSupport.decodeConfig(indexSchedule: """
        "index_topk_pattern": "FSFS"
        """)
        let routing = GLM4MoELiteRoutingPlan(config)
        let logits = MLXArray([Float(0), Float(4)]).reshaped(1, 1, 2)
        let bias = MLXArray([Float(10), Float(0)])
        let routed = routing.route(
            logits: logits,
            correctionBias: bias,
            outputDType: .float32
        )

        eval(routed.indices, routed.scores)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [0])
        #expect(abs(routed.scores.item(Float.self) - 0.5) < 0.0001)
    }

    @Test("router masks lower-scoring groups before expert selection")
    func routerMasksLowerScoringGroupsBeforeExpertSelection() throws {
        let config = try GLMMoEDSATestSupport.decodeConfig(
            indexSchedule: """
            "index_topk_pattern": "FSFS"
            """,
            nRoutedExperts: 4,
            nGroup: 2,
            topkGroup: 1
        )
        let routing = GLM4MoELiteRoutingPlan(config)
        let logits = MLXArray([Float(1), Float(4), Float(3), Float(2)]).reshaped(1, 1, 4)
        let routed = routing.route(
            logits: logits,
            correctionBias: MLXArray.zeros([4]),
            outputDType: .float32
        )

        eval(routed.indices)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [2])
    }

    @Test("constructs GLM DSA model with full and shared cache layout")
    func constructsGLMDSAModelWithCacheLayout() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = try GLMMoEDSATestSupport.writeModelDirectory(indexSchedule: """
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
            let loraTargets = model.loraLinearLayers()
            let _: any GreedyTokenModel = model

            #expect(cache.count == 4)
            #expect((cache[0] as? CacheList)?.layoutCaches.count == 2)
            #expect((cache[1] as? CacheList)?.layoutCaches.count == 1)
            #expect((cache[2] as? CacheList)?.layoutCaches.count == 2)
            #expect((cache[3] as? CacheList)?.layoutCaches.count == 1)
            #expect(loraTargets.count == 4)
            #expect(loraTargets[0].1 == ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"])
        }
    }

    @Test("sanitizer packs experts, splits KV projection, and strips tied heads")
    func sanitizerPacksExpertsSplitsKVProjectionAndStripsTiedHeads() throws {
        let sanitized = try GLMMoEDSATestSupport.sanitizedTiedCheckpointWeights()

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.4.self_attn.q_proj.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.kv_b_proj.weight"] == nil)

        let gate = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        let embedQ = try #require(sanitized["model.layers.0.self_attn.embed_q.weight"])
        let unembedOut = try #require(sanitized["model.layers.0.self_attn.unembed_out.weight"])

        eval(gate, embedQ, unembedOut)

        #expect(gate.shape == [2, 2, 2])
        #expect(embedQ.shape == [2, 4, 4])
        #expect(unembedOut.shape == [2, 4, 4])
        #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
        #expect(embedQ.asArray(Float.self).allSatisfy { $0 == 1 })
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
            let config = try GLMMoEDSATestSupport.decodeConfig(indexSchedule: """
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
}
