@testable import MLXLocalModels
import Testing

@Suite("MLX runtime memory guard limit diagnostics")
struct MLXRuntimeMemoryGuardLimitTests {
    @Test("guard identifies Metal ceiling when it is tighter than process memory")
    func guardIdentifiesMetalCeiling() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try Self.expectMetalMemoryGuardRejects()
        }
        let snapshots = Self.memoryGuardSnapshots(from: recorded.events)
        let rejected = try #require(snapshots.last)

        #expect(rejected.stage == .rejected)
        #expect(rejected.tier == .balanced)
        #expect(rejected.limitBytes == 1_000)
        #expect(rejected.limitSource == .metalRecommendedWorkingSet)
        #expect(rejected.message?.contains("Metal recommended working set ceiling") == true)
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

    private static func expectMetalMemoryGuardRejects() throws {
        do {
            try MLXRuntimeMemoryGuard.preflight(
                configuration: MLXMemoryGuardConfiguration(tier: .balanced, hardLimitFraction: 1),
                profile: Self.smallProfile,
                promptTokenCount: 128,
                cachedTokenCount: 0,
                prefillStepSize: 64,
                currentMemoryBytes: 900,
                cacheMemoryBytes: 0,
                physicalMemoryBytes: 32 * 1_073_741_824,
                metalLimitBytes: 1_000
            )
            Issue.record("Expected the Metal memory ceiling to reject the request")
        } catch LLMError.invalidConfiguration(let message) {
            #expect(message.contains("Metal recommended working set ceiling"))
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
