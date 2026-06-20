@testable import MLXLocalModels
import Testing

@Suite("MLX runtime memory guard MLA layout")
struct MLXRuntimeMemoryGuardMLATests {
    @Test("profile refines GLM DSA bytes from live cache layout")
    func profileRefinesGLMDSABytesFromLiveCacheLayout() throws {
        let profile = try #require(MLXModelMemoryProfile.profile(from: [
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "hidden_size": 256,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 4,
            "index_head_dim": 6
        ]))
        let liveCache: [KVCache] = [
            CacheList(KVCacheSimple(), KVCacheSimple()),
            CacheList(KVCacheSimple()),
            CacheList(KVCacheSimple(), KVCacheSimple()),
            CacheList(KVCacheSimple())
        ]
        let baseCacheBytes = 4 * (16 + 4) * 2
        let liveIndexerBytes = 2 * 6 * 2

        #expect(profile.estimatePromptKVBytes(tokenCount: 1) == Int64(baseCacheBytes))

        let refined = profile.refinedWithLiveCacheLayout(liveCache)

        #expect(refined.estimatePromptKVBytes(tokenCount: 1) == Int64(baseCacheBytes + liveIndexerBytes))
    }
}
