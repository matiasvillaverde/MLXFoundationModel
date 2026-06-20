import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model profile runtime optimization")
struct MLXModelProfileRuntimeOptimizationTests {
    @Test("DeepSeek V3.2 profile provides IndexCache runtime default")
    func deepSeekV32ProfileProvidesIndexCacheRuntimeDefault() {
        let optimization = Self.deepSeekV32Profile.optimizationProfile

        #expect(optimization.supportsIndexCache)
        #expect(optimization.defaultIndexCacheFrequency == 2)
        #expect(optimization.implementedFeatures.contains(.indexCache))
    }

    @Test("language model applies profile IndexCache default")
    func languageModelAppliesProfileIndexCacheDefault() {
        let runtime = ModelRuntimePreferences(
            promptCachePolicy: .memory,
            promptCacheByteLimit: 16_384
        )
        let languageModel = MLXLanguageModel(
            model: Self.model(profile: Self.deepSeekV32Profile),
            runtime: runtime
        )

        #expect(languageModel.runtime.optimization.mode == .off)
        #expect(languageModel.runtime.optimization.indexCacheFrequency == 2)
        #expect(languageModel.runtime.promptCachePolicy == .memory)
        #expect(languageModel.runtime.promptCacheByteLimit == 16_384)
    }

    @Test("language model preserves explicit IndexCache runtime frequency")
    func languageModelPreservesExplicitIndexCacheRuntimeFrequency() {
        let runtime = ModelRuntimePreferences(
            optimization: .indexCache(frequency: 4)
        )
        let languageModel = MLXLanguageModel(
            model: Self.model(profile: Self.deepSeekV32Profile),
            runtime: runtime
        )

        #expect(languageModel.runtime.optimization.indexCacheFrequency == 4)
    }

    @Test("DeepSeek V3 profile does not opt into V3.2 IndexCache default")
    func deepSeekV3ProfileDoesNotOptIntoV32IndexCacheDefault() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64
            ]
        )

        #expect(profile.optimizationProfile.supportsIndexCache)
        #expect(profile.optimizationProfile.defaultIndexCacheFrequency == nil)
    }

    @Test("MiniMax M3 disables TurboQuant KV until sparse cache quantization is implemented")
    func miniMaxM3DisablesTurboQuantKVUntilSparseCacheQuantizationIsImplemented() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "minimax_m3",
                "architectures": ["MiniMaxM3ForCausalLM"]
            ]
        )
        let optimization = profile.optimizationProfile

        #expect(optimization.promptCacheReuseAlignment == .prefillStep)
        #expect(optimization.supportsIndexCache)
        #expect(optimization.implementedFeatures.contains(.indexCache))
        #expect(!optimization.supportsTurboQuantKV)
        #expect(!optimization.implementedFeatures.contains(.turboQuantKV))
        #expect(optimization.status(for: .turboQuantKV) == nil)
    }

    @Test("MiniMax M3 VL also disables TurboQuant KV")
    func miniMaxM3VLAlsoDisablesTurboQuantKV() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "minimax_m3_vl",
                "architectures": ["MiniMaxM3VLForConditionalGeneration"],
                "vision_config": ["hidden_size": 1_280],
                "text_config": ["hidden_size": 6_144]
            ]
        )

        #expect(profile.optimizationProfile.promptCacheReuseAlignment == .prefillStep)
        #expect(profile.optimizationProfile.supportsIndexCache)
        #expect(!profile.optimizationProfile.supportsTurboQuantKV)
        #expect(!profile.optimizationProfile.detectedFeatures.contains(.turboQuantKV))
    }

    @Test("ordinary profiles keep TurboQuant KV available")
    func ordinaryProfilesKeepTurboQuantKVAvailable() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"]
            ]
        )

        #expect(profile.optimizationProfile.supportsTurboQuantKV)
        #expect(profile.optimizationProfile.implementedFeatures.contains(.turboQuantKV))
    }

    private static var deepSeekV32Profile: MLXModelProfile {
        MLXModelProfile.make(
            config: [
                "model_type": "deepseek_v32",
                "architectures": ["DeepseekV32ForCausalLM"],
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64
            ]
        )
    }

    private static func model(profile: MLXModelProfile) -> MLXModel {
        MLXModel(
            id: "deepseek-v32-fixture",
            location: URL(fileURLWithPath: "/tmp/deepseek-v32-fixture"),
            profile: profile
        )
    }
}
