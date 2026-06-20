import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX prompt cache observability", .serialized)
struct MLXPromptCacheObservabilityTests {
    @Test("records cumulative prefix cache hit and miss counters")
    func recordsCumulativePrefixCacheHitAndMissCounters() throws {
        let tracker = MLXPromptCacheObservabilityTracker(minSnapshotInterval: 0)

        _ = tracker.recordPlan(promptTokenCount: 11, reusedTokenCount: 0, timestamp: 0)
        let snapshot = tracker.recordPlan(
            promptTokenCount: 11,
            reusedTokenCount: 5,
            timestamp: 60
        )

        #expect(snapshot.counters.prefixHits == 1)
        #expect(snapshot.counters.prefixMisses == 1)
        #expect(snapshot.counters.prefixTokensRequested == 20)
        #expect(snapshot.counters.prefixTokensMatched == 5)
        #expect(snapshot.counters.prefixTokensSaved == 5)
        #expect(snapshot.counters.prefixHitRate == 0.5)
        #expect(snapshot.counters.prefixMatchEfficiency == 0.25)
    }

    @Test("computes rolling prefix rates from deltas")
    func computesRollingPrefixRatesFromDeltas() throws {
        let tracker = MLXPromptCacheObservabilityTracker(minSnapshotInterval: 0)

        _ = tracker.recordPlan(promptTokenCount: 11, reusedTokenCount: 0, timestamp: 0)
        let snapshot = tracker.recordPlan(
            promptTokenCount: 11,
            reusedTokenCount: 5,
            timestamp: 60
        )

        let oneMinute = try #require(snapshot.windows["1m"])
        #expect(oneMinute.prefixHits == 1)
        #expect(oneMinute.prefixMisses == 0)
        #expect(oneMinute.prefixHitRate == 1)
        #expect(oneMinute.prefixTokensRequested == 10)
        #expect(oneMinute.prefixTokensMatched == 5)
        #expect(oneMinute.prefixMatchEfficiency == 0.5)
    }

    @Test("tracks SSD and hot cache counters for future paged cache wiring")
    func tracksSSDAndHotCacheCountersForFuturePagedCacheWiring() throws {
        let tracker = MLXPromptCacheObservabilityTracker(minSnapshotInterval: 0)

        _ = tracker.recordSSDDiskLoad(timestamp: 4)
        _ = tracker.recordSSDHotHit(timestamp: 60)
        _ = tracker.recordSSDSave(timestamp: 61)
        _ = tracker.recordEviction(timestamp: 62)
        _ = tracker.recordHotCacheEviction(timestamp: 63)
        let snapshot = tracker.recordHotCachePromotion(timestamp: 64)

        #expect(snapshot.counters.ssdDiskLoads == 1)
        #expect(snapshot.counters.ssdHotHits == 1)
        #expect(snapshot.counters.ssdSaves == 1)
        #expect(snapshot.counters.evictions == 1)
        #expect(snapshot.counters.hotCacheEvictions == 1)
        #expect(snapshot.counters.hotCachePromotions == 1)
        #expect(snapshot.counters.ssdHotRate == 0.5)

        let oneMinute = try #require(snapshot.windows["1m"])
        #expect(oneMinute.ssdDiskLoads == 0)
        #expect(oneMinute.ssdHotHits == 1)
        #expect(oneMinute.ssdHotRate == 1)
        #expect(oneMinute.evictions == 1)
    }

    @Test("records prompt cache observability diagnostics")
    func recordsPromptCacheObservabilityDiagnostics() async throws {
        try await PromptCacheTestIsolation.withLock {
            MLXGenerationDiagnostics.resetPromptCacheObservability()
            defer { MLXGenerationDiagnostics.resetPromptCacheObservability() }

            let recorded = try await MLXGenerationDiagnostics.withRecording {
                MLXGenerationDiagnostics.recordPromptCachePlan(
                    promptTokenCount: 11,
                    reusedTokenCount: 5
                )
            }

            let snapshots = Self.observabilitySnapshots(from: recorded.events)
            let snapshot = try #require(snapshots.last)

            #expect(snapshot.counters.prefixHits == 1)
            #expect(snapshot.counters.prefixMisses == 0)
            #expect(snapshot.counters.prefixTokensRequested == 10)
            #expect(snapshot.counters.prefixTokensSaved == 5)
            #expect(
                MLXGenerationDiagnostics.promptCacheObservabilitySnapshot()
                    .counters.prefixTokensSaved == 5
            )
        }
    }

    @Test("persistent block store records SSD saves and budget evictions")
    func persistentBlockStoreRecordsSSDSavesAndBudgetEvictions() async throws {
        try await PromptCacheTestIsolation.withLock {
            MLXGenerationDiagnostics.resetPromptCacheObservability()
            defer { MLXGenerationDiagnostics.resetPromptCacheObservability() }
            let root = try Self.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }

            let recorded = try await MLXGenerationDiagnostics.withRecording {
                try Self.storeTwoPersistentBlocksAndPrune(root: root)
            }

            let snapshots = Self.observabilitySnapshots(from: recorded.events)
            let first = try #require(snapshots.first)
            let last = try #require(snapshots.last)

            #expect(last.counters.ssdSaves - first.counters.ssdSaves >= 1)
            #expect(last.counters.evictions - first.counters.evictions >= 1)
        }
    }

    @Test("persistent payload cache promotes disk loads and serves hot hits")
    func persistentPayloadCachePromotesDiskLoadsAndServesHotHits() async throws {
        try await PromptCacheTestIsolation.withLock {
            try await Self.withIsolatedHotCache(limitBytes: 128) {
                let record = try Self.storePersistentBlockOutsideRecording()
                defer { try? FileManager.default.removeItem(at: record.root) }
                MLXPersistentPromptCacheBlockStore.clearHotCache()

                let recorded = try await MLXGenerationDiagnostics.withRecording {
                    let url = MLXPersistentPromptCacheBlockStore.dataURL(
                        for: record.record,
                        rootURL: record.root
                    )
                    _ = try MLXPersistentPromptCacheBlockStore.loadPayload(at: url)
                    _ = try MLXPersistentPromptCacheBlockStore.loadPayload(at: url)
                }

                let counters = try Self.lastObservabilityCounters(from: recorded.events)
                #expect(counters.ssdDiskLoads >= 1)
                #expect(counters.ssdHotHits >= 1)
                #expect(counters.hotCachePromotions >= 1)
            }
        }
    }

    @Test("persistent hot payload cache evicts least recently used payloads")
    func persistentHotPayloadCacheEvictsLeastRecentlyUsedPayloads() async throws {
        try await PromptCacheTestIsolation.withLock {
            try await Self.withIsolatedHotCache(limitBytes: 12) {
                let root = try Self.makeTemporaryDirectory()
                defer { try? FileManager.default.removeItem(at: root) }

                let recorded = try await MLXGenerationDiagnostics.withRecording {
                    _ = try Self.storeBlock(hash: "1", bytes: 8, root: root)
                    _ = try Self.storeBlock(hash: "2", bytes: 8, root: root)
                }

                let counters = try Self.lastObservabilityCounters(from: recorded.events)
                let hotSnapshot = MLXPersistentPromptCacheBlockStore.hotCacheSnapshot()
                #expect(counters.hotCacheEvictions == 1)
                #expect(hotSnapshot.entryCount == 1)
                #expect(hotSnapshot.totalBytes == 8)
            }
        }
    }

    @Test("budget pruning removes hot payload entries")
    func budgetPruningRemovesHotPayloadEntries() async throws {
        try await PromptCacheTestIsolation.withLock {
            try await Self.withIsolatedHotCache(limitBytes: 128) {
                let stored = try Self.storePersistentBlockOutsideRecording()
                let url = MLXPersistentPromptCacheBlockStore.dataURL(
                    for: stored.record,
                    rootURL: stored.root
                )

                try MLXPersistentPromptCacheBlockStore.enforceBudget(
                    rootURL: stored.root,
                    limitBytes: 0
                )

                do {
                    _ = try MLXPersistentPromptCacheBlockStore.loadPayload(at: url)
                    Issue.record("Pruned payload should not be served from the hot cache")
                } catch {
                    // Expected: both SSD and hot payload entries were pruned.
                }
                try? FileManager.default.removeItem(at: stored.root)
            }
        }
    }

    private static func observabilitySnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPromptCacheObservabilitySnapshot] {
        events.compactMap { event in
            guard case .promptCacheObservability(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func lastObservabilityCounters(
        from events: [MLXGenerationDiagnosticEvent]
    ) throws -> MLXPromptCacheObservabilityCounters {
        try #require(observabilitySnapshots(from: events).last).counters
    }

    private struct StoredBlock {
        let root: URL
        let record: MLXPersistentPromptCacheBlockRecord
    }

    private static func withIsolatedHotCache<T>(
        limitBytes: Int,
        operation: () async throws -> T
    ) async throws -> T {
        MLXGenerationDiagnostics.resetPromptCacheObservability()
        MLXPersistentPromptCacheBlockStore.clearHotCache()
        MLXPersistentPromptCacheBlockStore.configureHotCache(limitBytes: limitBytes)
        defer {
            MLXPersistentPromptCacheBlockStore.clearHotCache()
            MLXPersistentPromptCacheBlockStore.configureHotCache(limitBytes: 67_108_864)
            MLXGenerationDiagnostics.resetPromptCacheObservability()
        }
        return try await operation()
    }

    private static func storePersistentBlockOutsideRecording() throws -> StoredBlock {
        let root = try Self.makeTemporaryDirectory()
        let record = try Self.storeBlock(hash: "a", bytes: 8, root: root)
        MLXGenerationDiagnostics.resetPromptCacheObservability()
        return StoredBlock(root: root, record: record)
    }

    private static func storeTwoPersistentBlocksAndPrune(root: URL) throws {
        _ = try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash("1"),
            tokenCount: 10,
            signature: signature(),
            payload: Data(repeating: 1, count: 10),
            rootURL: root,
            now: date(1)
        )
        _ = try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash("2"),
            tokenCount: 10,
            signature: signature(),
            payload: Data(repeating: 2, count: 10),
            rootURL: root,
            now: date(2)
        )
        try MLXPersistentPromptCacheBlockStore.enforceBudget(rootURL: root, limitBytes: 10)
    }

    private static func storeBlock(
        hash character: Character,
        bytes: Int,
        root: URL
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        let byte = character.unicodeScalars.first.map { UInt8($0.value) } ?? 0
        return try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash(character),
            tokenCount: bytes,
            signature: signature(),
            payload: Data(repeating: byte, count: bytes),
            rootURL: root,
            now: date(TimeInterval(bytes))
        )
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return root
    }

    private static func signature() -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters())
    }

    private static func hash(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private static func date(_ offset: TimeInterval) -> Date {
        Date(timeIntervalSinceReferenceDate: offset)
    }
}
