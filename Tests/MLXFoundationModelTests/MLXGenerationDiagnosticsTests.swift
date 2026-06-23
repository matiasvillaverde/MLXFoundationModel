@testable import MLXLocalModels
import Testing

@Suite("MLX generation diagnostics")
struct MLXGenerationDiagnosticsTests {
    @Test("cache snapshots are opt-in inside diagnostic recordings")
    func cacheSnapshotsAreOptInInsideDiagnosticRecordings() async throws {
        let skipped = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordCacheSnapshot(label: "step", cache: [])
        }

        #expect(Self.cacheSnapshotLabels(from: skipped.events).isEmpty)

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            await MLXGenerationDiagnostics.withCacheSnapshotRecording {
                MLXGenerationDiagnostics.recordCacheSnapshot(label: "step", cache: [])
            }
        }

        #expect(Self.cacheSnapshotLabels(from: recorded.events) == ["step"])
    }

    private static func cacheSnapshotLabels(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .cacheSnapshot(let snapshot) = event else {
                return nil
            }
            return snapshot.label
        }
    }
}
