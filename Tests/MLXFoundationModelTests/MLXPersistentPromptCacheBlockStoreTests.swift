import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache block store")
struct MLXPersistentPromptCacheBlockStoreTests {
    @Test("stores block payloads in hash-sharded files and scans metadata")
    func storesBlockPayloadsInHashShardedFilesAndScansMetadata() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let hash = Self.hash("a")

        let record = try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash,
            tokenCount: 256,
            signature: Self.signature(),
            payload: Data(repeating: 7, count: 12),
            rootURL: root,
            now: Self.date(10)
        )
        let dataURL = MLXPersistentPromptCacheBlockStore.dataURL(for: record, rootURL: root)

        #expect(record.byteCount == 12)
        #expect(dataURL.deletingLastPathComponent().lastPathComponent == "aa")
        #expect(FileManager.default.fileExists(atPath: dataURL.path))
        #expect(try MLXPersistentPromptCacheBlockStore.scan(rootURL: root) == [record])
    }

    @Test("lookup touches metadata and leaves missing blocks as misses")
    func lookupTouchesMetadataAndLeavesMissingBlocksAsMisses() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let hash = Self.hash("b")

        _ = try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash,
            tokenCount: 128,
            signature: Self.signature(),
            payload: Data(repeating: 1, count: 8),
            rootURL: root,
            now: Self.date(1)
        )

        let lookup = try MLXPersistentPromptCacheBlockStore.lookup(
            blockHash: hash,
            rootURL: root,
            now: Self.date(20)
        )
        let touched = try #require(lookup)

        #expect(touched.lastAccess == Self.date(20))
        #expect(try MLXPersistentPromptCacheBlockStore.lookup(
            blockHash: Self.hash("c"),
            rootURL: root
        ) == nil)
    }

    @Test("scan removes stale metadata and orphan data files")
    func scanRemovesStaleMetadataAndOrphanDataFiles() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let staleMetadataHash = Self.hash("d")
        let orphanDataHash = Self.hash("e")

        let staleRecord = try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: staleMetadataHash,
            tokenCount: 1,
            signature: Self.signature(),
            payload: Data(repeating: 1, count: 1),
            rootURL: root
        )
        try FileManager.default.removeItem(
            at: MLXPersistentPromptCacheBlockStore.dataURL(for: staleRecord, rootURL: root)
        )

        let orphanRecord = try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: orphanDataHash,
            tokenCount: 1,
            signature: Self.signature(),
            payload: Data(repeating: 2, count: 1),
            rootURL: root
        )
        try FileManager.default.removeItem(
            at: MLXPersistentPromptCacheBlockStore.metadataURL(for: orphanRecord, rootURL: root)
        )

        #expect(try MLXPersistentPromptCacheBlockStore.scan(rootURL: root).isEmpty)
        #expect(!FileManager.default.fileExists(atPath: MLXPersistentPromptCacheBlockStore
            .metadataURL(for: staleRecord, rootURL: root).path))
        #expect(!FileManager.default.fileExists(atPath: MLXPersistentPromptCacheBlockStore
            .dataURL(for: orphanRecord, rootURL: root).path))
    }

    @Test("parent block budget ignores nested segment store")
    func parentBlockBudgetIgnoresNestedSegmentStore() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let segmentRoot = root.appendingPathComponent("IndependentBlocks", isDirectory: true)

        let block = try Self.store(hash: Self.hash("a"), bytes: 10, access: 1, root: root)
        let segment = try Self.store(hash: Self.hash("b"), bytes: 10, access: 2, root: segmentRoot)

        try MLXPersistentPromptCacheBlockStore.enforceBudget(rootURL: root, limitBytes: 0)

        #expect(try MLXPersistentPromptCacheBlockStore.scan(rootURL: root).isEmpty)
        #expect(FileManager.default.fileExists(
            atPath: MLXPersistentPromptCacheBlockStore.dataURL(for: segment, rootURL: segmentRoot).path
        ))
        #expect(!FileManager.default.fileExists(
            atPath: MLXPersistentPromptCacheBlockStore.dataURL(for: block, rootURL: root).path
        ))
    }

    @Test("budget enforcement prunes least recently used unprotected blocks")
    func budgetEnforcementPrunesLeastRecentlyUsedUnprotectedBlocks() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let oldHash = Self.hash("1")
        let protectedHash = Self.hash("2")
        let recentHash = Self.hash("3")

        _ = try Self.store(hash: oldHash, bytes: 10, access: 1, root: root)
        _ = try Self.store(hash: protectedHash, bytes: 20, access: 2, root: root)
        _ = try Self.store(hash: recentHash, bytes: 20, access: 3, root: root)

        try MLXPersistentPromptCacheBlockStore.enforceBudget(
            rootURL: root,
            limitBytes: 25,
            protectedBlockHashes: [protectedHash]
        )

        let remainingHashes = try MLXPersistentPromptCacheBlockStore
            .scan(rootURL: root)
            .map(\.blockHash)
        #expect(remainingHashes == [protectedHash])
    }

    @Test("storage-hash protection does not keep incompatible records for the same block")
    func storageHashProtectionDoesNotKeepIncompatibleRecordsForSameBlock() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let sharedHash = Self.hash("4")

        let incompatible = try Self.store(
            hash: sharedHash,
            bytes: 10,
            access: 1,
            root: root,
            signature: Self.signature(prefillStepSize: 8)
        )
        let protected = try Self.store(
            hash: sharedHash,
            bytes: 20,
            access: 2,
            root: root,
            signature: Self.signature(prefillStepSize: 16)
        )

        #expect(incompatible.storageHash != protected.storageHash)

        try MLXPersistentPromptCacheBlockStore.enforceBudget(
            rootURL: root,
            limitBytes: 25,
            protectedStorageHashes: [protected.storageHash]
        )

        let remaining = try MLXPersistentPromptCacheBlockStore.scan(rootURL: root)
        #expect(remaining.map(\.storageHash) == [protected.storageHash])
        Self.assertRemoved(incompatible, andKept: protected, root: root)
    }

    private static func assertRemoved(
        _ removed: MLXPersistentPromptCacheBlockRecord,
        andKept kept: MLXPersistentPromptCacheBlockRecord,
        root: URL
    ) {
        #expect(!FileManager.default.fileExists(
            atPath: MLXPersistentPromptCacheBlockStore.dataURL(for: removed, rootURL: root).path
        ))
        #expect(FileManager.default.fileExists(
            atPath: MLXPersistentPromptCacheBlockStore.dataURL(for: kept, rootURL: root).path
        ))
    }

    private static func store(
        hash: String,
        bytes: Int,
        access: TimeInterval,
        root: URL,
        signature: PromptCacheSignature = signature()
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash,
            tokenCount: bytes,
            signature: signature,
            payload: Data(repeating: 0, count: bytes),
            rootURL: root,
            now: date(access)
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

    private static func signature(
        prefillStepSize: Int = GenerationConstants.defaultPrefillStepSize
    ) -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters(prefillStepSize: prefillStepSize))
    }

    private static func hash(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private static func date(_ offset: TimeInterval) -> Date {
        Date(timeIntervalSinceReferenceDate: offset)
    }
}
