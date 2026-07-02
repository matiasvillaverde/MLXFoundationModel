import Foundation
@testable import MLXLocalModels
import Testing

extension MLXRealModelPersistentPromptCacheTests {
    static func promptCachePlans(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPromptCachePlanSnapshot] {
        events.compactMap { event in
            guard case .promptCachePlan(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    static func promptCacheLookups(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPromptCacheLookupSnapshot] {
        events.compactMap { event in
            guard case .promptCacheLookup(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    static func lastPromptCacheCounters(
        from events: [MLXGenerationDiagnosticEvent]
    ) throws -> MLXPromptCacheObservabilityCounters {
        try #require(events.compactMap { event in
            guard case .promptCacheObservability(let snapshot) = event else {
                return nil
            }
            return snapshot.counters
        }.last)
    }

    static func cacheSummary(
        _ events: [MLXGenerationDiagnosticEvent],
        result: MLXRealModelHarness.GenerationResult,
        modelID: String,
        phase: String
    ) -> String {
        [
            "modelID=\(modelID)",
            "phase=\(phase)",
            "textPreview=\(result.text.prefix(160))",
            "textChunkCount=\(result.textChunkCount)",
            "generatedTokens=\(result.metrics?.usage?.generatedTokens ?? -1)",
            "stopReason=\(String(describing: result.metrics?.generation?.stopReason))",
            "usageReused=\(result.metrics?.usage?.promptCacheReusedTokenCount ?? -1)",
            "plans=\(promptCachePlans(from: events))",
            "lookups=\(promptCacheLookups(from: events))",
            "lifecycle=\(result.lifecycleEvents)"
        ].joined(separator: "\n")
    }

    static func removePersistentArtifacts(
        identity: PromptCacheIdentity,
        configuration: ProviderConfiguration
    ) {
        try? FileManager.default.removeItem(at: MLXPersistentPromptCacheStore.url(for: configuration))
        removePersistentRecords(identity: identity, root: MLXPersistentPromptCacheBlockStore.rootURL())
        removePersistentRecords(identity: identity, root: MLXPersistentPromptCacheSegmentStore.rootURL())
        MLXPersistentPromptCacheBlockStore.clearHotCache()
    }

    private static func removePersistentRecords(
        identity: PromptCacheIdentity,
        root: URL
    ) {
        guard let records = try? MLXPersistentPromptCacheBlockStore.scan(rootURL: root) else {
            return
        }
        for record in records where record.signature.promptCacheIdentity == identity {
            try? FileManager.default.removeItem(
                at: MLXPersistentPromptCacheBlockStore.dataURL(for: record, rootURL: root)
            )
            try? FileManager.default.removeItem(
                at: MLXPersistentPromptCacheBlockStore.metadataURL(for: record, rootURL: root)
            )
        }
    }
}
