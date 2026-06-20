import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache snapshot store")
struct MLXPromptCacheSnapshotStoreTests {
    @Test("stores and restores the deepest block-aligned prefix snapshot")
    func storesAndRestoresDeepestBlockAlignedPrefixSnapshot() async throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tokenIds = Array(0 ..< 10)
        let signature = Self.signature()
        let entry = Self.entry(tokens: tokenIds, signature: signature)

        let record = try #require(try MLXPersistentPromptCacheSnapshotStore.storeSnapshot(
            entry: entry,
            blockSize: Self.blockSize,
            rootURL: root,
            now: Self.date(1),
            encoder: Self.encode
        ))
        let recorded = try await Self.restoreSnapshotWithRecording(
            tokenIds: Array(0 ..< 12),
            signature: signature,
            root: root
        )
        let restored = try #require(recorded.result)

        #expect(record.tokenCount == 8)
        #expect(restored.tokens == Array(0 ..< 8))
        #expect(restored.signature == signature)
        try Self.expectPersistentSnapshotLookup(in: recorded.events)
    }

    @Test("restore ignores incompatible signatures")
    func restoreIgnoresIncompatibleSignatures() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tokenIds = Array(0 ..< 8)
        let entry = Self.entry(tokens: tokenIds, signature: Self.signature(kvBits: 4))

        try MLXPersistentPromptCacheSnapshotStore.storeSnapshot(
            entry: entry,
            blockSize: Self.blockSize,
            rootURL: root,
            encoder: Self.encode
        )
        let restored = try MLXPersistentPromptCacheSnapshotStore.restoreBestSnapshot(
            tokenIds: tokenIds,
            signature: Self.signature(),
            blockSize: Self.blockSize,
            maxBytes: nil,
            rootURL: root,
            loader: Self.load
        )

        #expect(restored == nil)
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

    private static func signature(kvBits: Int? = nil) -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters(kvBits: kvBits))
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

    private static func date(_ offset: TimeInterval) -> Date {
        Date(timeIntervalSinceReferenceDate: offset)
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

    private static func restoreSnapshotWithRecording(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        root: URL
    ) async throws -> (result: PromptCacheEntry?, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            try MLXPersistentPromptCacheSnapshotStore.restoreBestSnapshot(
                tokenIds: tokenIds,
                signature: signature,
                blockSize: Self.blockSize,
                maxBytes: nil,
                rootURL: root,
                now: Self.date(2),
                loader: Self.load
            )
        }
    }

    private static func expectPersistentSnapshotLookup(
        in events: [MLXGenerationDiagnosticEvent]
    ) throws {
        let lookup = try #require(Self.lookupSnapshots(from: events).last)
        #expect(lookup.strategy == .persistentSnapshot)
        #expect(lookup.matchedBlockCount == 2)
        #expect(lookup.candidateCount == 1)
        #expect(lookup.reusedTokenCount == 8)
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
