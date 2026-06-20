import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache segment alignment")
struct MLXPromptCacheSegmentStoreAlignmentTests {
    @Test("prefill-step reuse restores only aligned partial segments")
    func prefillStepReuseRestoresOnlyAlignedPartialSegments() async throws {
        try await Self.withDefaultHotCache {
            let root = try Self.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let signature = Self.signature(prefillStepSize: 8)
            let entry = Self.entry(tokens: Array(0 ..< 12), signature: signature)

            try MLXPersistentPromptCacheSegmentStore.storeSegments(
                entry: entry,
                blockSize: Self.blockSize,
                rootURL: root,
                encoder: Self.encode
            )
            let recorded = try await Self.restoreSegmentsWithRecording(
                tokenIds: Array(0 ..< 16),
                signature: signature,
                root: root,
                reusePolicy: PromptCacheReusePolicy(alignment: .prefillStep, prefillStepSize: 8)
            )

            #expect(recorded.result.map(\.blockIndex) == [0, 1])
            #expect(recorded.result.flatMap(\.tokens) == Array(0 ..< 8))
            try Self.expectPersistentSegmentLookup(
                in: recorded.events,
                reusedTokenCount: 8
            )
        }
    }

    @Test("prefill-step reuse skips partial segments below one step")
    func prefillStepReuseSkipsPartialSegmentsBelowOneStep() async throws {
        try await Self.withDefaultHotCache {
            let root = try Self.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let signature = Self.signature(prefillStepSize: 8)
            let entry = Self.entry(tokens: Array(0 ..< 4), signature: signature)

            try MLXPersistentPromptCacheSegmentStore.storeSegments(
                entry: entry,
                blockSize: Self.blockSize,
                rootURL: root,
                encoder: Self.encode
            )
            let recorded = try await Self.restoreSegmentsWithRecording(
                tokenIds: Array(0 ..< 8),
                signature: signature,
                root: root,
                reusePolicy: PromptCacheReusePolicy(alignment: .prefillStep, prefillStepSize: 8)
            )

            #expect(recorded.result.isEmpty)
            try Self.expectPersistentSegmentLookup(
                in: recorded.events,
                reusedTokenCount: 0,
                matchedBlockCount: 1,
                candidateCount: 1
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

    private static func signature(prefillStepSize: Int) -> PromptCacheSignature {
        PromptCacheSignature(
            parameters: GenerateParameters(prefillStepSize: prefillStepSize)
        )
    }

    private static func restoreSegmentsWithRecording(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        root: URL,
        reusePolicy: PromptCacheReusePolicy
    ) async throws -> (result: [MLXPersistentPromptCacheSegment], events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            try MLXPersistentPromptCacheSegmentStore.restorePrefixSegments(
                tokenIds: tokenIds,
                signature: signature,
                blockSize: Self.blockSize,
                reusePolicy: reusePolicy,
                rootURL: root,
                loader: Self.load
            )
        }
    }

    private static func expectPersistentSegmentLookup(
        in events: [MLXGenerationDiagnosticEvent],
        reusedTokenCount: Int,
        matchedBlockCount: Int = 2,
        candidateCount: Int = 2
    ) throws {
        let lookup = try #require(Self.lookupSnapshots(from: events).last)
        #expect(lookup.strategy == .persistentSegments)
        #expect(lookup.matchedBlockCount == matchedBlockCount)
        #expect(lookup.candidateCount == candidateCount)
        #expect(lookup.reusedTokenCount == reusedTokenCount)
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
