import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache segment store")
struct MLXPromptCacheSegmentStoreTests {
    @Test("stores and restores independent block segments")
    func storesAndRestoresIndependentBlockSegments() async throws {
        try await Self.withDefaultHotCache {
            let root = try Self.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let tokenIds = Array(0 ..< 10)
            let signature = Self.signature()
            let entry = Self.entry(tokens: tokenIds, signature: signature)

            let records = try MLXPersistentPromptCacheSegmentStore.storeSegments(
                entry: entry,
                blockSize: Self.blockSize,
                rootURL: root,
                encoder: Self.encode
            )
            let recorded = try await Self.restoreSegmentsWithRecording(
                tokenIds: Array(0 ..< 12),
                signature: signature,
                root: root
            )
            let segments = recorded.result

            try Self.expectRestoredPrefix(records: records, segments: segments, events: recorded.events)
        }
    }

    @Test("restores independent segments as a prompt cache entry")
    func restoresIndependentSegmentsAsPromptCacheEntry() async throws {
        try await Self.withDefaultHotCache {
            let root = try Self.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let tokenIds = Array(0 ..< 10)
            let signature = Self.signature()
            let entry = Self.entry(tokens: tokenIds, signature: signature)

            try MLXPersistentPromptCacheSegmentStore.storeSegments(
                entry: entry,
                blockSize: Self.blockSize,
                rootURL: root,
                encoder: Self.encode
            )
            let restored = try #require(try MLXPersistentPromptCacheSegmentStore.restoreBestEntry(
                tokenIds: Array(0 ..< 12),
                signature: signature,
                blockSize: Self.blockSize,
                maxBytes: nil,
                rootURL: root,
                loader: Self.load
            ))

            #expect(restored.tokens == Array(0 ..< 8))
            #expect(restored.signature == signature)
            #expect(restored.cache.isEmpty)
        }
    }

    @Test("assembly rejects non-contiguous segment indexes")
    func assemblyRejectsNonContiguousSegmentIndexes() {
        let signature = Self.signature()
        let segment = MLXPersistentPromptCacheSegment(
            blockHash: "abcd",
            blockIndex: 1,
            tokens: Array(4 ..< 8),
            cache: [],
            signature: signature,
            dataURL: URL(fileURLWithPath: "/tmp/segment.safetensors")
        )

        let entry = MLXPersistentPromptCacheSegmentStore.assembleEntry(
            from: [segment],
            maxBytes: nil
        )

        #expect(entry == nil)
    }

    @Test("restore fails closed when segment metadata is missing")
    func restoreFailsClosedWhenSegmentMetadataIsMissing() async throws {
        try await Self.withDefaultHotCache {
            let root = try Self.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let tokenIds = Array(0 ..< 8)
            let signature = Self.signature()
            let entry = Self.entry(tokens: tokenIds, signature: signature)
            let records = try MLXPersistentPromptCacheSegmentStore.storeSegments(
                entry: entry,
                blockSize: Self.blockSize,
                rootURL: root,
                encoder: Self.encode
            )
            let firstRecord = try #require(records.first)
            let firstURL = MLXPersistentPromptCacheBlockStore.dataURL(
                for: firstRecord,
                rootURL: root
            )
            try Data("{}".utf8).write(to: firstURL, options: .atomic)

            let recorded = try await Self.restoreSegmentsWithRecording(
                tokenIds: tokenIds,
                signature: signature,
                root: root
            )
            let segments = recorded.result

            #expect(segments.isEmpty)
            try Self.expectPersistentSegmentLookup(
                in: recorded.events,
                reusedTokenCount: 0
            )
        }
    }

    private static let blockSize = 4

    private static func entry(
        tokens: [Int],
        signature: PromptCacheSignature
    ) -> PromptCacheEntry {
        PromptCacheEntry(
            tokens: tokens,
            cache: [],
            signature: signature,
            byteCount: 0
        )
    }

    private static func signature(
        prefillStepSize: Int = GenerationConstants.defaultPrefillStepSize
    ) -> PromptCacheSignature {
        PromptCacheSignature(
            parameters: GenerateParameters(prefillStepSize: prefillStepSize)
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

    private static func encode(
        _: [KVCache],
        _ metadata: [String: String]
    ) throws -> Data {
        try JSONEncoder().encode(metadata)
    }

    private static func load(_ url: URL) throws -> ([KVCache], [String: String]) {
        let data = try Data(contentsOf: url)
        return ([], try JSONDecoder().decode([String: String].self, from: data))
    }

    private static func withDefaultHotCache<T>(
        _ operation: () async throws -> T
    ) async throws -> T {
        try await PromptCacheTestIsolation.withLock {
            PromptCacheTestIsolation.resetSharedHotCache()
            defer { PromptCacheTestIsolation.resetSharedHotCache() }
            return try await operation()
        }
    }

    private static func restoreSegmentsWithRecording(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        root: URL
    ) async throws -> (result: [MLXPersistentPromptCacheSegment], events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            try MLXPersistentPromptCacheSegmentStore.restorePrefixSegments(
                tokenIds: tokenIds,
                signature: signature,
                blockSize: Self.blockSize,
                rootURL: root,
                loader: Self.load
            )
        }
    }

    private static func expectPersistentSegmentLookup(
        in events: [MLXGenerationDiagnosticEvent],
        reusedTokenCount: Int
    ) throws {
        let lookup = try #require(Self.lookupSnapshots(from: events).last)
        #expect(lookup.strategy == .persistentSegments)
        #expect(lookup.matchedBlockCount == 2)
        #expect(lookup.candidateCount == 2)
        #expect(lookup.reusedTokenCount == reusedTokenCount)
    }

    private static func expectRestoredPrefix(
        records: [MLXPersistentPromptCacheBlockRecord],
        segments: [MLXPersistentPromptCacheSegment],
        events: [MLXGenerationDiagnosticEvent]
    ) throws {
        #expect(records.count == 2)
        #expect(records.allSatisfy { record in
            record.tokenCount == Self.blockSize
        })
        #expect(segments.map(\.blockIndex) == [0, 1])
        #expect(segments.map(\.tokens) == [
            Array(0 ..< 4),
            Array(4 ..< 8)
        ])
        try Self.expectPersistentSegmentLookup(
            in: events,
            reusedTokenCount: 8
        )
    }

    private static func lookupSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPromptCacheLookupSnapshot] {
        events.compactMap { event in
            guard case .promptCacheLookup(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}
