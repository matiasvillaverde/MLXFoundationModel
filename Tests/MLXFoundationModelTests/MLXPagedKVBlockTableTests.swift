@testable import MLXLocalModels
import Testing

@Suite("MLX paged KV block table")
struct MLXPagedKVBlockTableTests {
    @Test("shares prefix blocks and protects retained pages")
    func sharesPrefixBlocksAndProtectsRetainedPages() throws {
        var table = MLXPagedKVBlockTable(capacity: 2)
        let first = try table.allocate(blockHash: "a", tokenCount: 4)
        let second = try table.allocate(blockHash: "b", tokenCount: 4)
        try table.release(first)
        try table.release(second)

        try table.attachPrefix(rowID: 1, blockIDs: [first, second])
        try table.attachPrefix(rowID: 2, blockIDs: [first, second])

        #expect(Self.record(first, in: table).refCount == 2)
        #expect(Self.record(second, in: table).refCount == 2)
        #expect(table.attachedBlockIDs(for: 1) == [first, second])

        do {
            _ = try table.allocate(blockHash: "c", tokenCount: 4)
            Issue.record("Expected retained blocks to be protected from eviction")
        } catch MLXPagedKVBlockTableError.noEvictableBlock(let capacity) {
            #expect(capacity == 2)
        }
    }

    @Test("forks shared block before write")
    func forksSharedBlockBeforeWrite() throws {
        var table = MLXPagedKVBlockTable(capacity: 2)
        let shared = try table.allocate(blockHash: "prefix", tokenCount: 8)
        try table.retain(shared)

        let forked = try table.forkForWrite(
            shared,
            blockHash: "prefix-plus-token",
            tokenCount: 9
        )

        #expect(forked != shared)
        #expect(Self.record(shared, in: table).refCount == 1)
        #expect(Self.record(shared, in: table).blockHash == "prefix")
        #expect(Self.record(forked, in: table).refCount == 1)
        #expect(Self.record(forked, in: table).blockHash == "prefix-plus-token")
        #expect(Self.record(forked, in: table).tokenCount == 9)
    }

    @Test("evicts least recently used unretained block")
    func evictsLeastRecentlyUsedUnretainedBlock() throws {
        var table = MLXPagedKVBlockTable(capacity: 2)
        let first = try table.allocate(blockHash: "first", tokenCount: 4)
        let second = try table.allocate(blockHash: "second", tokenCount: 4)
        try table.release(first)
        try table.release(second)
        try table.touch(second)

        let reused = try table.allocate(blockHash: "third", tokenCount: 4)

        #expect(reused == first)
        #expect(table.record(for: first)?.blockHash == "third")
        #expect(table.record(for: second)?.blockHash == "second")
        #expect(table.blockIDs(matchingHash: "first").isEmpty)
    }

    @Test("detach releases row prefix leases")
    func detachReleasesRowPrefixLeases() throws {
        var table = MLXPagedKVBlockTable(capacity: 2)
        let first = try table.allocate(blockHash: "a", tokenCount: 4)
        let second = try table.allocate(blockHash: "b", tokenCount: 4)
        try table.release(first)
        try table.release(second)
        try table.attachPrefix(rowID: 9, blockIDs: [first, second])

        let detached = try table.detach(rowID: 9)

        #expect(detached == [first, second])
        #expect(table.attachedBlockIDs(for: 9).isEmpty)
        #expect(Self.record(first, in: table).refCount == 0)
        #expect(Self.record(second, in: table).refCount == 0)
    }

    @Test("records block table diagnostics")
    func recordsBlockTableDiagnostics() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            var table = MLXPagedKVBlockTable(capacity: 1)
            let first = try table.allocate(blockHash: "a", tokenCount: 4)
            try table.release(first)
            _ = try table.allocate(blockHash: "b", tokenCount: 4)
        }

        let snapshots = recorded.events.compactMap { event -> MLXPagedKVBlockTableSnapshot? in
            guard case .pagedKVBlocks(let snapshot) = event else {
                return nil
            }
            return snapshot
        }

        #expect(snapshots.map(\.stage) == [.allocated, .released, .evicted, .allocated])
        #expect(snapshots.last?.capacity == 1)
        #expect(snapshots.last?.usedCount == 1)
        #expect(snapshots.last?.freeCount == 0)
        #expect(snapshots.last?.refCounts == [1])
    }

    @Test("stores chained prompt-cache block hashes")
    func storesChainedPromptCacheBlockHashes() throws {
        let tokenIDs = Array(0 ..< 12)
        let hashes = PromptCacheBlockIndex.prefixBlockHashes(for: tokenIDs, blockSize: 4)
        var table = MLXPagedKVBlockTable(capacity: hashes.count)

        for hash in hashes {
            let id = try table.allocate(blockHash: hash, tokenCount: 4)
            try table.release(id)
        }

        #expect(table.orderedRecords.map(\.blockHash) == hashes)
        #expect(table.blockIDs(matchingHash: hashes[1]).count == 1)
    }

    private static func record(
        _ id: MLXPagedKVBlockID,
        in table: MLXPagedKVBlockTable
    ) -> MLXPagedKVBlockRecord {
        guard let record = table.record(for: id) else {
            Issue.record("Missing block \(id)")
            return .init(
                id: id,
                blockHash: "",
                tokenCount: 0,
                refCount: 0,
                lastAccessTick: 0
            )
        }
        return record
    }
}
