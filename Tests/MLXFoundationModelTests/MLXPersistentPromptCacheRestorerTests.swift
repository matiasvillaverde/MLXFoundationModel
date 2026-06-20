import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache restorer")
struct MLXPersistentPromptCacheRestorerTests {
    @Test("chooses deeper prefix snapshot over shallower independent segments")
    func choosesDeeperPrefixSnapshotOverShallowerSegments() async throws {
        let roots = try Self.makeTemporaryRoots()
        defer { Self.removeTemporaryRoots(roots) }
        let signature = Self.signature()

        try Self.storeSegments(tokens: Array(0 ..< 8), signature: signature, roots: roots)
        try Self.storeSnapshot(tokens: Array(0 ..< 16), signature: signature, roots: roots)

        let recorded = try await Self.restoreWithRecording(
            tokenIds: Array(0 ..< 18),
            signature: signature,
            roots: roots
        )
        let restored = try #require(recorded.result)

        #expect(restored.tokens == Array(0 ..< 16))
        try Self.expectLookupStrategies(in: recorded.events, reusedCounts: [
            .persistentSegments: 8,
            .persistentSnapshot: 16
        ])
    }

    @Test("chooses deeper independent segments over shallower prefix snapshot")
    func choosesDeeperSegmentsOverShallowerSnapshot() async throws {
        let roots = try Self.makeTemporaryRoots()
        defer { Self.removeTemporaryRoots(roots) }
        let signature = Self.signature()

        try Self.storeSegments(tokens: Array(0 ..< 16), signature: signature, roots: roots)
        try Self.storeSnapshot(tokens: Array(0 ..< 8), signature: signature, roots: roots)

        let recorded = try await Self.restoreWithRecording(
            tokenIds: Array(0 ..< 18),
            signature: signature,
            roots: roots
        )
        let restored = try #require(recorded.result)

        #expect(restored.tokens == Array(0 ..< 16))
        try Self.expectLookupStrategies(in: recorded.events, reusedCounts: [
            .persistentSegments: 16,
            .persistentSnapshot: 8
        ])
    }

    @Test("falls back to prefix snapshot when segment payload is corrupt")
    func fallsBackToSnapshotWhenSegmentPayloadIsCorrupt() async throws {
        let roots = try Self.makeTemporaryRoots()
        defer { Self.removeTemporaryRoots(roots) }
        let signature = Self.signature()

        let records = try Self.storeSegments(tokens: Array(0 ..< 16), signature: signature, roots: roots)
        try Self.corruptFirstSegment(records: records, roots: roots)
        try Self.storeSnapshot(tokens: Array(0 ..< 8), signature: signature, roots: roots)

        let recorded = try await Self.restoreWithRecording(
            tokenIds: Array(0 ..< 18),
            signature: signature,
            roots: roots
        )
        let restored = try #require(recorded.result)

        #expect(restored.tokens == Array(0 ..< 8))
        try Self.expectLookupStrategies(in: recorded.events, reusedCounts: [
            .persistentSegments: 0,
            .persistentSnapshot: 8
        ])
    }

    private static let blockSize = 4

    private struct Roots {
        let block: URL
        let segment: URL
    }

    @discardableResult
    private static func storeSegments(
        tokens: [Int],
        signature: PromptCacheSignature,
        roots: Roots
    ) throws -> [MLXPersistentPromptCacheBlockRecord] {
        try MLXPersistentPromptCacheSegmentStore.storeSegments(
            entry: entry(tokens: tokens, signature: signature),
            blockSize: blockSize,
            rootURL: roots.segment,
            encoder: encode
        )
    }

    private static func storeSnapshot(
        tokens: [Int],
        signature: PromptCacheSignature,
        roots: Roots
    ) throws {
        try MLXPersistentPromptCacheSnapshotStore.storeSnapshot(
            entry: entry(tokens: tokens, signature: signature),
            blockSize: blockSize,
            rootURL: roots.block,
            encoder: encode
        )
    }

    private static func corruptFirstSegment(
        records: [MLXPersistentPromptCacheBlockRecord],
        roots: Roots
    ) throws {
        let first = try #require(records.first)
        let url = MLXPersistentPromptCacheBlockStore.dataURL(for: first, rootURL: roots.segment)
        try Data("not-json".utf8).write(to: url, options: .atomic)
    }

    private static func restoreWithRecording(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        roots: Roots
    ) async throws -> (result: PromptCacheEntry?, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            MLXPersistentPromptCacheRestorer.restoreBestEntry(
                tokenIds: tokenIds,
                signature: signature,
                blockSize: blockSize,
                maxBytes: nil,
                blockRootURL: roots.block,
                segmentRootURL: roots.segment,
                segmentLoader: load,
                snapshotLoader: load
            )
        }
    }

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

    private static func signature() -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters())
    }

    private static func makeTemporaryRoots() throws -> Roots {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let roots = Roots(
            block: root.appendingPathComponent("blocks", isDirectory: true),
            segment: root.appendingPathComponent("segments", isDirectory: true)
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return roots
    }

    private static func removeTemporaryRoots(_ roots: Roots) {
        let root = roots.block.deletingLastPathComponent()
        try? FileManager.default.removeItem(at: root)
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

    private static func expectLookupStrategies(
        in events: [MLXGenerationDiagnosticEvent],
        reusedCounts: [MLXPromptCacheLookupSnapshot.Strategy: Int]
    ) throws {
        let lookups = lookupSnapshots(from: events)
        for (strategy, reusedCount) in reusedCounts {
            let lookup = try #require(lookups.first { $0.strategy == strategy })
            #expect(lookup.reusedTokenCount == reusedCount)
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
