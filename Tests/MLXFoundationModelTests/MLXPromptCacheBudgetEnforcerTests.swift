import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache budget enforcer")
struct MLXPromptCacheBudgetEnforcerTests {
    @Test("enforces one LRU budget across snapshot block and segment stores")
    func enforcesOneLRUBudgetAcrossSnapshotBlockAndSegmentStores() throws {
        let roots = try Self.makeRoots()
        defer { try? FileManager.default.removeItem(at: roots.root) }

        let oldSnapshot = try Self.writeSnapshot(
            named: "old.safetensors",
            bytes: 10,
            access: 1,
            root: roots.snapshot
        )
        let block = try Self.storeBlock(hash: "b", bytes: 20, access: 2, root: roots.block)
        let segment = try Self.storeBlock(hash: "c", bytes: 20, access: 3, root: roots.segment)
        let recentSnapshot = try Self.writeSnapshot(
            named: "recent.safetensors",
            bytes: 30,
            access: 4,
            root: roots.snapshot
        )

        try MLXPersistentPromptCacheBudgetEnforcer.enforceAll(
            limitBytes: 50,
            snapshotRootURL: roots.snapshot,
            blockRootURL: roots.block,
            segmentRootURL: roots.segment
        )

        #expect(!FileManager.default.fileExists(atPath: oldSnapshot.path))
        #expect(!Self.exists(block, root: roots.block))
        #expect(Self.exists(segment, root: roots.segment))
        #expect(FileManager.default.fileExists(atPath: recentSnapshot.path))
    }

    @Test("reserves incoming bytes before admitting new SSD cache blocks")
    func reservesIncomingBytesBeforeAdmittingNewSSDCacheBlocks() throws {
        let roots = try Self.makeRoots()
        defer { try? FileManager.default.removeItem(at: roots.root) }

        let oldSnapshot = try Self.writeSnapshot(
            named: "old.safetensors",
            bytes: 10,
            access: 1,
            root: roots.snapshot
        )
        let oldBlock = try Self.storeBlock(hash: "b", bytes: 20, access: 2, root: roots.block)
        let keptSegment = try Self.storeBlock(hash: "c", bytes: 20, access: 3, root: roots.segment)

        try MLXPersistentPromptCacheBudgetEnforcer.enforceAllBeforeInsert(
            limitBytes: 50,
            incomingByteCount: 25,
            snapshotRootURL: roots.snapshot,
            blockRootURL: roots.block,
            segmentRootURL: roots.segment
        )
        let incoming = try Self.storeBlock(hash: "d", bytes: 25, access: 4, root: roots.block)

        #expect(!FileManager.default.fileExists(atPath: oldSnapshot.path))
        #expect(!Self.exists(oldBlock, root: roots.block))
        #expect(Self.exists(keptSegment, root: roots.segment))
        #expect(Self.exists(incoming, root: roots.block))
        #expect(try Self.totalBytes(roots: roots) <= 50)
    }

    private struct Roots {
        let root: URL
        let snapshot: URL
        let block: URL
        let segment: URL
    }

    private static func makeRoots() throws -> Roots {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshot = root.appendingPathComponent("Snapshots", isDirectory: true)
        let block = root.appendingPathComponent("Blocks", isDirectory: true)
        let segment = block.appendingPathComponent("IndependentBlocks", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return Roots(root: root, snapshot: snapshot, block: block, segment: segment)
    }

    private static func writeSnapshot(
        named name: String,
        bytes: Int,
        access: TimeInterval,
        root: URL
    ) throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var url = root.appendingPathComponent(name)
        try Data(repeating: 0, count: bytes).write(to: url)
        try setAccess(access, for: &url)
        return url
    }

    private static func storeBlock(
        hash character: Character,
        bytes: Int,
        access: TimeInterval,
        root: URL
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash(character),
            tokenCount: bytes,
            signature: signature(),
            payload: Data(repeating: 0, count: bytes),
            rootURL: root,
            now: date(access)
        )
    }

    private static func exists(
        _ record: MLXPersistentPromptCacheBlockRecord,
        root: URL
    ) -> Bool {
        FileManager.default.fileExists(
            atPath: MLXPersistentPromptCacheBlockStore.dataURL(for: record, rootURL: root).path
        )
    }

    private static func totalBytes(roots: Roots) throws -> Int {
        let snapshotFiles = try MLXPersistentPromptCacheStore.cacheFiles(
            rootURL: roots.snapshot,
            fileManager: .default
        )
        let blockRecords = try MLXPersistentPromptCacheBlockStore.scan(rootURL: roots.block)
        let segmentRecords = try MLXPersistentPromptCacheBlockStore.scan(rootURL: roots.segment)
        let snapshots = snapshotFiles.reduce(0) { total, file in
            total + file.byteCount
        }
        let blocks = blockRecords.reduce(0) { total, record in
            total + record.byteCount
        }
        let segments = segmentRecords.reduce(0) { total, record in
            total + record.byteCount
        }
        return snapshots + blocks + segments
    }

    private static func setAccess(_ access: TimeInterval, for url: inout URL) throws {
        var values = URLResourceValues()
        let date = date(access)
        values.contentAccessDate = date
        values.contentModificationDate = date
        try url.setResourceValues(values)
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
