@testable import MLXLocalModels
import Testing

extension MLXRealModelHarness {
    static func verifyPromptCacheProgress(
        _ result: GenerationResult,
        reusedTokenCount: Int,
        summary: Comment
    ) throws {
        let progress = try #require(promptProgressEvent(from: result.lifecycleEvents))
        let reusedUnits = Int64(reusedTokenCount)
        #expect(progress.completedUnitCount == reusedUnits, summary)
        #expect(progress.cachedUnitCount == reusedUnits, summary)
        #expect((progress.totalUnitCount ?? 0) >= reusedUnits, summary)
    }

    private static func promptProgressEvent(
        from events: [StreamLifecycleEvent]
    ) -> StreamLifecycleEvent? {
        events.last { event in
            event.phase == .promptProcessing && event.state == .progress
        }
    }
}
