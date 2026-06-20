import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX runtime memory guard load preflight")
struct MLXRuntimeMemoryGuardLoadTests {
    @Test("model load estimator counts only model artifacts")
    func modelLoadEstimatorCountsOnlyModelArtifacts() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(repeating: 0, count: 128).write(to: root.appendingPathComponent("config.json"))
        try Data(repeating: 0, count: 1_024).write(to: root.appendingPathComponent("weights.safetensors"))
        try Data(repeating: 0, count: 2_048).write(to: root.appendingPathComponent("model-00002.bin"))
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("nested"),
            withIntermediateDirectories: true
        )
        try Data(repeating: 0, count: 512)
            .write(to: root.appendingPathComponent("nested/adapter.npz"))

        let bytes = try MLXRuntimeMemoryGuard.estimatedModelLoadBytes(modelDirectory: root)

        #expect(bytes == 3_584)
    }

    @Test("model load preflight rejects before MLX load and records diagnostics")
    func modelLoadPreflightRejectsBeforeMLXLoad() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            do {
                try MLXRuntimeMemoryGuard.preflightModelLoad(
                    configuration: MLXMemoryGuardConfiguration(
                        tier: .custom,
                        customLimitBytes: 1_000,
                        hardLimitFraction: 1
                    ),
                    modelLoadBytes: 400,
                    currentMemoryBytes: 700,
                    cacheMemoryBytes: 0,
                    physicalMemoryBytes: 32 * 1_073_741_824,
                    metalLimitBytes: nil
                )
                Issue.record("Expected the model load guard to reject the request")
            } catch LLMError.invalidConfiguration(let message) {
                #expect(message.contains("Model load would require"))
                #expect(message.contains("model weights"))
            }
        }
        let rejected = try #require(Self.memoryGuardSnapshots(from: recorded.events).last)

        #expect(rejected.stage == .modelLoadRejected)
        #expect(rejected.currentMemoryBytes == 700)
        #expect(rejected.estimatedPeakBytes == 400)
        #expect(rejected.limitBytes == 1_000)
        #expect(rejected.limitSource == .customLimit)
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXRuntimeMemoryGuardTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
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
