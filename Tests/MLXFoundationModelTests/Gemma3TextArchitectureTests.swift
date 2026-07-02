import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Gemma 3 text architecture")
struct Gemma3TextArchitectureTests {
    @Test("runtime max KV size bounds global and sliding caches")
    func runtimeMaxKVSizeBoundsGlobalAndSlidingCaches() throws {
        let model = Gemma3TextModel(try Self.decodeConfig())
        let cache = model.newCache(parameters: GenerateParameters(maxKVSize: 8))

        #expect(cache.count == 4)
        #expect(cache.allSatisfy { $0.maxSize == 8 })
        #expect(cache.allSatisfy { $0 is RotatingKVCache })
    }

    @Test("default cache keeps global layers dense and local layers sliding")
    func defaultCacheKeepsGlobalLayersDenseAndLocalLayersSliding() throws {
        let model = Gemma3TextModel(try Self.decodeConfig())
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 4)
        #expect(cache[0].maxSize == 16)
        #expect(cache[1].maxSize == nil)
        #expect(cache[2].maxSize == 16)
        #expect(cache[3].maxSize == nil)
        #expect(cache[0] is RotatingKVCache)
        #expect(cache[1] is KVCacheSimple)
        #expect(cache[2] is RotatingKVCache)
        #expect(cache[3] is KVCacheSimple)
    }

    private static func decodeConfig() throws -> Gemma3TextConfiguration {
        let fields = [
            #""model_type": "gemma3_text""#,
            #""hidden_size": 16"#,
            #""num_hidden_layers": 4"#,
            #""intermediate_size": 32"#,
            #""num_attention_heads": 2"#,
            #""head_dim": 8"#,
            #""rms_norm_eps": 0.000001"#,
            #""vocab_size": 64"#,
            #""num_key_value_heads": 1"#,
            #""rope_global_base_freq": 1000000.0"#,
            #""rope_local_base_freq": 10000.0"#,
            #""rope_traditional": false"#,
            #""query_pre_attn_scalar": 8"#,
            #""sliding_window": 16"#,
            #""sliding_window_pattern": 2"#
        ]
        let json = "{\n  \(fields.joined(separator: ",\n  "))\n}"
        return try JSONDecoder().decode(
            Gemma3TextConfiguration.self,
            from: Data(json.utf8)
        )
    }
}
