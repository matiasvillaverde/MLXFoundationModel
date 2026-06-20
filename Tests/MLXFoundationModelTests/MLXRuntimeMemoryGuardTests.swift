@testable import MLXLocalModels
import Testing

@Suite("MLX runtime memory guard")
struct MLXRuntimeMemoryGuardTests {
    @Test("memory profile estimates fused prefill peaks for arbitrary head dimensions")
    func memoryProfileEstimatesPrefillPeaks() {
        let fusedPeak = Self.fusedProfile.estimatePrefillPeakBytes(
            newTokenCount: 32_768,
            cachedTokenCount: 0,
            prefillStepSize: 2_048
        )
        let fusedKV = Int64(62 * 4 * 128 * 2 * 2 * 32_768)
        let fusedSDPA = Int64(32 * 2_048 * 128 * 4)

        let unusualHeadDimensionPeak = Self.unusualHeadDimensionProfile.estimatePrefillPeakBytes(
            newTokenCount: 32_768,
            cachedTokenCount: 0,
            prefillStepSize: 2_048
        )
        let unusualKV = Int64(48 * 4 * 256 * 2 * 2 * 32_768)
        let unusualOutput = Int64(8 * 2_048 * 256 * 4)

        #expect(fusedPeak == fusedKV + fusedSDPA)
        #expect(unusualHeadDimensionPeak == unusualKV + unusualOutput)
    }

    @Test("memory profile keeps short unusual heads on fused SDPA path")
    func memoryProfileKeepsShortUnusualHeadsOnFusedSDPAPath() {
        let peak = Self.unusualHeadDimensionProfile.estimatePrefillPeakBytes(
            newTokenCount: 128,
            cachedTokenCount: 0,
            prefillStepSize: 8
        )
        let expectedKV = Int64(48 * 4 * 256 * 2 * 2 * 128)
        let expectedOutput = Int64(8 * 8 * 256 * 4)

        #expect(peak == expectedKV + expectedOutput)
    }

    @Test("profile loader uses MLA latent KV dimensions")
    func profileLoaderUsesMLALatentKVDimensions() throws {
        let profile = try #require(MLXModelMemoryProfile.profile(from: [
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "hidden_size": 8_192,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64
        ]))

        #expect(profile.headDimension == 128)
        #expect(profile.estimatePromptKVBytes(tokenCount: 1) == Int64(78 * (512 + 64) * 2))
    }

    @Test("profile loader includes GLM DSA indexer cache dimensions")
    func profileLoaderIncludesGLMDSAIndexerCacheDimensions() throws {
        let profile = try #require(MLXModelMemoryProfile.profile(from: [
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "hidden_size": 256,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 4,
            "index_head_dim": 6,
            "indexer_types": ["full", "shared", "full", "shared"]
        ]))
        let mainCacheBytes = 4 * (16 + 4) * 2
        let indexerCacheBytes = 2 * 6 * 2

        #expect(profile.headDimension == 32)
        #expect(profile.estimatePromptKVBytes(tokenCount: 1) == Int64(mainCacheBytes + indexerCacheBytes))
    }

    @Test("profile loader derives GLM DSA indexer pattern when explicit types are absent")
    func profileLoaderDerivesGLMDSAIndexerPatternWhenExplicitTypesAreAbsent() throws {
        let profile = try #require(MLXModelMemoryProfile.profile(from: [
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 5,
            "num_attention_heads": 4,
            "hidden_size": 128,
            "kv_lora_rank": 8,
            "qk_rope_head_dim": 4,
            "index_head_dim": 3,
            "index_topk": 64,
            "index_topk_pattern": "FSSFS"
        ]))
        let mainCacheBytes = 5 * (8 + 4) * 2
        let indexerCacheBytes = 2 * 3 * 2

        #expect(profile.estimatePromptKVBytes(tokenCount: 1) == Int64(mainCacheBytes + indexerCacheBytes))
    }

    @Test("profile loader prefers nested language model config for multimodal packs")
    func profileLoaderPrefersNestedLanguageModelConfigForMultimodalPacks() throws {
        let profile = try #require(MLXModelMemoryProfile.profile(from: [
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "hidden_size": 1_024,
            "vision_config": ["num_hidden_layers": 27],
            "text_config": [
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 64,
                "kv_lora_rank": 8,
                "qk_rope_head_dim": 4
            ]
        ]))

        #expect(profile.numLayers == 2)
        #expect(profile.numAttentionHeads == 4)
        #expect(profile.numKVHeads == 2)
        #expect(profile.headDimension == 16)
        #expect(profile.estimatePromptKVBytes(tokenCount: 1) == Int64(2 * (8 + 4) * 2))
    }

    @Test("TurboQuant KV bits reduce KV estimate without reducing SDPA activation estimate")
    func turboQuantKVBitsReduceKVEstimateWithoutReducingSDPAActivationEstimate() {
        let turboProfile = Self.fusedProfile.applyingKVCacheBits(2.5)
        let kvBytes = turboProfile.estimatePromptKVBytes(tokenCount: 128)
        let prefillPeak = turboProfile.estimatePrefillPeakBytes(
            newTokenCount: 128,
            cachedTokenCount: 0,
            prefillStepSize: 64
        )
        let expectedKV = Int64((Double(62 * 4 * 128 * 2 * 128) * 2.5 / 8.0).rounded(.up))
        let expectedSDPA = Int64(32 * 64 * 128 * 4)

        #expect(kvBytes == expectedKV)
        #expect(prefillPeak == expectedKV + expectedSDPA)
    }

    @Test("TurboQuant KV bits preserve fused SDPA activation estimate for unusual heads")
    func turboQuantKVBitsPreserveFusedSDPAActivationEstimateForUnusualHeads() {
        let turboProfile = Self.unusualHeadDimensionProfile.applyingKVCacheBits(2.5)
        let peak = turboProfile.estimatePrefillPeakBytes(
            newTokenCount: 128,
            cachedTokenCount: 256,
            prefillStepSize: 64
        )
        let expectedKV = Int64((Double(48 * 4 * 256 * 2 * 128) * 2.5 / 8.0).rounded(.up))
        let expectedOutput = Int64(8 * 64 * 256 * 4)

        #expect(peak == expectedKV + expectedOutput)
    }

    @Test("guard rejects requests above the configured ceiling and records diagnostics")
    func guardRejectsRequestsAboveCeiling() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try Self.expectMemoryGuardRejects()
        }
        let snapshots = Self.memoryGuardSnapshots(from: recorded.events)
        let rejected = try #require(snapshots.last)

        #expect(rejected.stage == .rejected)
        #expect(rejected.tier == .custom)
        #expect(rejected.currentMemoryBytes == 900)
        #expect((rejected.estimatedPeakBytes ?? 0) > 0)
        #expect(rejected.limitBytes == 1_000)
        #expect(rejected.limitSource == .customLimit)
        #expect(rejected.message?.contains("custom memory guard limit") == true)
    }

    @Test("guard accounts for reused prompt cache tokens")
    func guardAccountsForReusedPromptCacheTokens() async throws {
        let profile = MLXModelMemoryProfile(
            numLayers: 1,
            numKVHeads: 1,
            numAttentionHeads: 1,
            headDimension: 64,
            dtypeSize: 2,
            scoreDTypeSize: 2,
            kvBytesPerTokenOverride: nil
        )
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try MLXRuntimeMemoryGuard.preflight(
                configuration: MLXMemoryGuardConfiguration(tier: .custom, customLimitBytes: 1_000_000),
                profile: profile,
                promptTokenCount: 100,
                cachedTokenCount: 90,
                prefillStepSize: 32,
                currentMemoryBytes: 0,
                cacheMemoryBytes: 0,
                physicalMemoryBytes: 32 * 1_073_741_824,
                metalLimitBytes: nil
            )
        }
        let snapshots = Self.memoryGuardSnapshots(from: recorded.events)
        let allowed = try #require(snapshots.last)

        #expect(allowed.stage == .allowed)
        #expect(allowed.promptTokenCount == 100)
        #expect(allowed.cachedTokenCount == 90)
        #expect(allowed.newTokenCount == 10)
    }

    private static var fusedProfile: MLXModelMemoryProfile {
        MLXModelMemoryProfile(
            numLayers: 62,
            numKVHeads: 4,
            numAttentionHeads: 32,
            headDimension: 128,
            dtypeSize: 2,
            scoreDTypeSize: 2,
            kvBytesPerTokenOverride: nil
        )
    }

    private static var unusualHeadDimensionProfile: MLXModelMemoryProfile {
        MLXModelMemoryProfile(
            numLayers: 48,
            numKVHeads: 4,
            numAttentionHeads: 8,
            headDimension: 256,
            dtypeSize: 2,
            scoreDTypeSize: 2,
            kvBytesPerTokenOverride: nil
        )
    }

    private static var smallProfile: MLXModelMemoryProfile {
        MLXModelMemoryProfile(
            numLayers: 4,
            numKVHeads: 2,
            numAttentionHeads: 2,
            headDimension: 128,
            dtypeSize: 2,
            scoreDTypeSize: 2,
            kvBytesPerTokenOverride: nil
        )
    }

    private static func expectMemoryGuardRejects() throws {
        do {
            try MLXRuntimeMemoryGuard.preflight(
                configuration: MLXMemoryGuardConfiguration(
                    tier: .custom,
                    customLimitBytes: 1_000,
                    hardLimitFraction: 1
                ),
                profile: Self.smallProfile,
                promptTokenCount: 128,
                cachedTokenCount: 0,
                prefillStepSize: 64,
                currentMemoryBytes: 900,
                cacheMemoryBytes: 0,
                physicalMemoryBytes: 32 * 1_073_741_824,
                metalLimitBytes: nil
            )
            Issue.record("Expected the memory guard to reject the request")
        } catch LLMError.invalidConfiguration(let message) {
            #expect(message.contains("Generation would require"))
        }
    }

    private static func memoryGuardSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXMemoryGuardSnapshot] {
        events.compactMap { event in
            guard case .memoryGuard(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}
