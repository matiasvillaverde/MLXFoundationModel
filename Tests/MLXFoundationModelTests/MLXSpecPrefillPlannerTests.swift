@testable import MLXLocalModels
import Testing

@Suite("MLX SpecPrefill planner")
struct MLXSpecPrefillPlannerTests {
    @Test("plans sparse prefill with protected and cached prefixes")
    func plansSparsePrefillWithProtectedAndCachedPrefixes() async throws {
        let importance = (0..<96).map { index in
            Float((32..<64).contains(index) ? 10 : 1)
        }

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXSpecPrefillPlanner.plan(
                promptTokenCount: importance.count,
                cachedTokenCount: 8,
                protectedPrefixTokenCount: 12,
                importance: importance,
                configuration: .init(keepRate: 0.25, thresholdTokens: 32)
            )
        }

        let plan = try #require(recorded.result)
        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(plan.retainedTokenIndices == Array(0..<12) + Array(32..<64))
        #expect(plan.newPrefillTokenIndices == Array(8..<12) + Array(32..<64))
        #expect(plan.keepRate == 0.25)
        #expect(plan.thresholdTokens == 32)
        #expect(plan.decodePositionOffset == 52)
        #expect(snapshot.stage == .planned)
        #expect(snapshot.retainedTokenCount == 44)
        #expect(snapshot.newPrefillTokenCount == 36)
        #expect(snapshot.decodePositionOffset == 52)
    }

    @Test("skips sparse prefill below configured threshold")
    func skipsSparsePrefillBelowConfiguredThreshold() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXSpecPrefillPlanner.plan(
                promptTokenCount: 16,
                cachedTokenCount: 0,
                protectedPrefixTokenCount: 0,
                importance: Array(repeating: 1, count: 16),
                configuration: .init(keepRate: 0.2, thresholdTokens: 64)
            )
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(recorded.result == nil)
        #expect(snapshot.stage == .skippedBelowThreshold)
        #expect(snapshot.promptTokenCount == 16)
        #expect(snapshot.thresholdTokens == 64)
        #expect(snapshot.message?.contains("below") == true)
    }

    @Test("skips sparse prefill when importance scores are stale")
    func skipsSparsePrefillWhenImportanceScoresAreStale() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXSpecPrefillPlanner.plan(
                promptTokenCount: 64,
                cachedTokenCount: 0,
                protectedPrefixTokenCount: 0,
                importance: Array(repeating: 1, count: 63),
                configuration: .init(keepRate: 0.2, thresholdTokens: 1)
            )
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(recorded.result == nil)
        #expect(snapshot.stage == .skippedImportanceMismatch)
        #expect(snapshot.promptTokenCount == 64)
        #expect(snapshot.message?.contains("Importance") == true)
    }

    @Test("skips sparse prefill when chunk selection keeps every uncached token")
    func skipsSparsePrefillWhenChunkSelectionKeepsEveryUncachedToken() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXSpecPrefillPlanner.plan(
                promptTokenCount: 16,
                cachedTokenCount: 0,
                protectedPrefixTokenCount: 0,
                importance: Array(repeating: 1, count: 16),
                configuration: .init(keepRate: 0.1, thresholdTokens: 1)
            )
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(recorded.result == nil)
        #expect(snapshot.stage == .skippedNoReduction)
        #expect(snapshot.retainedTokenCount == 16)
        #expect(snapshot.newPrefillTokenCount == 16)
        #expect(snapshot.decodePositionOffset == 0)
    }

    @Test("skips sparse prefill when configuration is absent")
    func skipsSparsePrefillWhenConfigurationIsAbsent() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXSpecPrefillPlanner.plan(
                promptTokenCount: 64,
                cachedTokenCount: 0,
                protectedPrefixTokenCount: 0,
                importance: Array(repeating: 1, count: 64),
                configuration: nil
            )
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(recorded.result == nil)
        #expect(snapshot.stage == .skippedDisabled)
    }

    @Test("records dense fallback when sparse runtime is unavailable")
    func recordsDenseFallbackWhenSparseRuntimeIsUnavailable() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXSpecPrefillPlanner.recordRuntimeUnavailable(
                promptTokenCount: 128,
                cachedTokenCount: 256,
                protectedPrefixTokenCount: 512,
                configuration: .init(keepRate: 0.25, thresholdTokens: 64)
            )
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(snapshot.stage == .skippedRuntimeUnavailable)
        #expect(snapshot.promptTokenCount == 128)
        #expect(snapshot.cachedTokenCount == 128)
        #expect(snapshot.protectedPrefixTokenCount == 128)
        #expect(snapshot.keepRate == 0.25)
        #expect(snapshot.thresholdTokens == 64)
        #expect(snapshot.chunkSize == MLXSpecPrefillPlanner.defaultChunkSize)
        #expect(snapshot.message?.contains("dense prefill") == true)
    }

    private static func snapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXSpecPrefillPlanSnapshot] {
        events.compactMap { event in
            guard case .specPrefillPlan(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}
