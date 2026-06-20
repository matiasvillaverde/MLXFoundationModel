@testable import MLXLocalModels
import Testing

@Suite("MLX runtime memory guard decode")
struct MLXRuntimeMemoryGuardDecodeTests {
    @Test("generation peak estimate includes maximum decode KV growth")
    func generationPeakEstimateIncludesMaximumDecodeKVGrowth() {
        let prefillOnly = Self.profile.estimateGenerationPeakBytes(
            promptTokenCount: 8,
            cachedTokenCount: 0,
            maximumGeneratedTokenCount: 0,
            prefillStepSize: 8
        )
        let withDecode = Self.profile.estimateGenerationPeakBytes(
            promptTokenCount: 8,
            cachedTokenCount: 0,
            maximumGeneratedTokenCount: 64,
            prefillStepSize: 8
        )

        #expect(withDecode > prefillOnly)
        #expect(withDecode >= Self.profile.estimatePromptKVBytes(tokenCount: 72))
    }

    @Test("guard rejects small prompts when maximum decode KV would exceed limit")
    func guardRejectsSmallPromptWhenMaximumDecodeKVWouldExceedLimit() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            do {
                try Self.preflightDecodeGrowthOverLimit()
                Issue.record("Expected the memory guard to reject decode KV growth")
            } catch LLMError.invalidConfiguration(let message) {
                #expect(message.contains("Generation would require"))
                #expect(message.contains("prompt/decode KV"))
            }
        }

        let rejected = try #require(Self.memoryGuardSnapshots(from: recorded.events).last)

        #expect(rejected.stage == .rejected)
        #expect(rejected.promptTokenCount == 8)
        #expect(rejected.cachedTokenCount == 0)
        #expect(rejected.newTokenCount == 8)
        #expect(rejected.maximumGeneratedTokenCount == 64)
        #expect(rejected.limitBytes == 80_000)
        #expect((rejected.estimatedPeakBytes ?? 0) > 80_000)
    }

    private static func preflightDecodeGrowthOverLimit() throws {
        try MLXRuntimeMemoryGuard.preflight(
            configuration: MLXMemoryGuardConfiguration(
                tier: .custom,
                customLimitBytes: 80_000,
                hardLimitFraction: 1
            ),
            profile: Self.profile,
            promptTokenCount: 8,
            cachedTokenCount: 0,
            maximumGeneratedTokenCount: 64,
            prefillStepSize: 8,
            currentMemoryBytes: 0,
            cacheMemoryBytes: 0,
            physicalMemoryBytes: 32 * 1_073_741_824,
            metalLimitBytes: nil
        )
    }

    private static var profile: MLXModelMemoryProfile {
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
