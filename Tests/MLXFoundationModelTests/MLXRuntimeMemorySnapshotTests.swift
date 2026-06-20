@testable import MLXLocalModels
import Testing

@Suite("MLX runtime memory snapshots")
struct MLXRuntimeMemorySnapshotTests {
    @Test("guard consumes injected memory snapshots")
    func guardConsumesInjectedMemorySnapshots() async throws {
        let snapshot = MLXRuntimeMemorySnapshot(
            currentMemoryBytes: 700,
            cacheMemoryBytes: 250,
            physicalMemoryBytes: 32 * 1_073_741_824,
            metalLimitBytes: nil
        )
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try MLXRuntimeMemoryGuard.preflight(
                configuration: MLXMemoryGuardConfiguration(
                    tier: .custom,
                    customLimitBytes: 1_000_000,
                    hardLimitFraction: 1,
                    includeCacheMemory: true
                ),
                profile: Self.smallProfile,
                promptTokenCount: 1,
                cachedTokenCount: 0,
                maximumGeneratedTokenCount: 0,
                prefillStepSize: 1,
                memorySnapshot: snapshot
            )
        }
        let allowed = try #require(Self.memoryGuardSnapshots(from: recorded.events).last)

        #expect(allowed.stage == .allowed)
        #expect(allowed.currentMemoryBytes == 950)
        #expect(allowed.limitBytes == 1_000_000)
        #expect(allowed.limitSource == .customLimit)
    }

    @Test("guard caps requests by host available memory")
    func guardCapsRequestsByHostAvailableMemory() async throws {
        let snapshot = MLXRuntimeMemorySnapshot(
            currentMemoryBytes: 900,
            cacheMemoryBytes: 0,
            physicalMemoryBytes: 32 * 1_073_741_824,
            metalLimitBytes: nil,
            availableMemoryBytes: 100
        )
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            do {
                try MLXRuntimeMemoryGuard.preflight(
                    configuration: MLXMemoryGuardConfiguration(tier: .balanced, hardLimitFraction: 1),
                    profile: Self.smallProfile,
                    promptTokenCount: 1,
                    cachedTokenCount: 0,
                    maximumGeneratedTokenCount: 0,
                    prefillStepSize: 1,
                    memorySnapshot: snapshot
                )
                Issue.record("Expected host available memory to reject the request")
            } catch LLMError.invalidConfiguration(let message) {
                #expect(message.contains("host available memory ceiling"))
            }
        }
        let rejected = try #require(Self.memoryGuardSnapshots(from: recorded.events).last)

        #expect(rejected.stage == .rejected)
        #expect(rejected.currentMemoryBytes == 900)
        #expect(rejected.limitBytes == 1_000)
        #expect(rejected.limitSource == .hostAvailableMemory)
    }

    @Test("live snapshot reports physical memory")
    func liveSnapshotReportsPhysicalMemory() {
        let snapshot = MLXRuntimeMemorySnapshot.live(metalLimitBytes: nil)

        #expect(snapshot.physicalMemoryBytes > 0)
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
