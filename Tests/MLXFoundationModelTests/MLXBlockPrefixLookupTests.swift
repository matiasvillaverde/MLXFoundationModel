import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache prefix lookup")
struct MLXBlockPrefixLookupTests {
    @Test("matches contiguous prefix blocks and touches records")
    func matchesContiguousPrefixBlocksAndTouchesRecords() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tokenIds = Array(0 ..< 12)
        let signature = Self.signature()
        let hashes = Self.hashes(for: tokenIds)
        try Self.store(hashes[0], signature: signature, root: root)
        try Self.store(hashes[1], signature: signature, root: root)

        let lookup = try MLXPersistentPromptCacheBlockStore.lookupPrefix(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: Self.blockSize,
            rootURL: root,
            now: Self.date(20)
        )
        let hit = try #require(lookup)

        #expect(hit.matchedBlockCount == 2)
        #expect(hit.cachedTokenCount == 8)
        #expect(hit.nextMissingBlockHash == hashes[2])
        #expect(hit.dataURLs == hit.records.map { record in
            MLXPersistentPromptCacheBlockStore.dataURL(for: record, rootURL: root)
        })
        #expect(hit.records.allSatisfy { record in
            record.lastAccess == Self.date(20)
        })
    }

    @Test("stops at the first missing block")
    func stopsAtTheFirstMissingBlock() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tokenIds = Array(0 ..< 12)
        let signature = Self.signature()
        let hashes = Self.hashes(for: tokenIds)
        try Self.store(hashes[0], signature: signature, root: root)
        try Self.store(hashes[2], signature: signature, root: root)

        let lookup = try MLXPersistentPromptCacheBlockStore.lookupPrefix(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: Self.blockSize,
            rootURL: root
        )
        let hit = try #require(lookup)

        #expect(hit.matchedBlockCount == 1)
        #expect(hit.cachedTokenCount == 4)
        #expect(hit.nextMissingBlockHash == hashes[1])
    }

    @Test("ignores signature mismatches without touching metadata")
    func ignoresSignatureMismatchesWithoutTouchingMetadata() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tokenIds = Array(0 ..< 8)
        let hash = Self.hashes(for: tokenIds)[0]
        try Self.store(hash, signature: Self.signature(kvBits: 4), root: root)

        let hit = try MLXPersistentPromptCacheBlockStore.lookupPrefix(
            tokenIds: tokenIds,
            signature: Self.signature(),
            blockSize: Self.blockSize,
            rootURL: root,
            now: Self.date(20)
        )
        let record = try #require(try MLXPersistentPromptCacheBlockStore.scan(rootURL: root).first)

        #expect(hit == nil)
        #expect(record.lastAccess == Self.date(1))
    }

    @Test("ignores block-size mismatches")
    func ignoresBlockSizeMismatches() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tokenIds = Array(0 ..< 8)
        let signature = Self.signature()
        let hash = Self.hashes(for: tokenIds)[0]
        try Self.store(hash, signature: signature, root: root, blockSize: 8)

        let hit = try MLXPersistentPromptCacheBlockStore.lookupPrefix(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: Self.blockSize,
            rootURL: root
        )

        #expect(hit == nil)
    }

    private static let blockSize = 4

    private static func store(
        _ hash: String,
        signature: PromptCacheSignature,
        root: URL,
        blockSize: Int = blockSize
    ) throws {
        try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash,
            blockSize: blockSize,
            tokenCount: blockSize,
            signature: signature,
            payload: Data(repeating: 0, count: blockSize),
            rootURL: root,
            now: date(1)
        )
    }

    private static func hashes(for tokenIds: [Int]) -> [String] {
        PromptCacheBlockIndex.prefixBlockHashes(for: tokenIds, blockSize: blockSize)
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

    private static func signature(kvBits: Int? = nil) -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters(kvBits: kvBits))
    }

    private static func date(_ offset: TimeInterval) -> Date {
        Date(timeIntervalSinceReferenceDate: offset)
    }
}
