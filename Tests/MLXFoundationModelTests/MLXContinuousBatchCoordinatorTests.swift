@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch coordinator")
struct MLXContinuousBatchCoordinatorTests {
    @Test("keeps processor sampler and cache state aligned across removal and late joins")
    func keepsRowStateAlignedAcrossRemovalAndLateJoins() throws {
        var coordinator = MLXContinuousBatchCoordinator<RowState>()
        let ids = try coordinator.admitBatch([
            .init(processor: "plain", sampler: "greedy", cache: "cache-a"),
            .init(processor: "grammar-json", sampler: "top-p", cache: "cache-b"),
            .init(processor: "thinking", sampler: "mirostat", cache: "cache-c")
        ])

        try coordinator.updateState(for: ids[1]) { state in
            state.generatedTokenCount = 7
        }
        _ = coordinator.finish(id: ids[0])
        let lateID = try coordinator.admit(
            .init(processor: "grammar-choice", sampler: "adaptive-p", cache: "cache-d")
        )
        try coordinator.realign(to: [ids[1], ids[2], lateID])

        #expect(coordinator.orderedRowIDs == [ids[1], ids[2], lateID])
        #expect(coordinator[ids[1]]?.processor == "grammar-json")
        #expect(coordinator[ids[1]]?.sampler == "top-p")
        #expect(coordinator[ids[1]]?.cache == "cache-b")
        #expect(coordinator[ids[1]]?.generatedTokenCount == 7)
        #expect(coordinator[ids[2]]?.processor == "thinking")
        #expect(coordinator[lateID]?.processor == "grammar-choice")
        #expect(coordinator[lateID]?.sampler == "adaptive-p")
    }

    @Test("rejects empty batch admission")
    func rejectsEmptyBatchAdmission() throws {
        var coordinator = MLXContinuousBatchCoordinator<RowState>()

        do {
            _ = try coordinator.admitBatch([])
            Issue.record("Expected empty batch admission to fail")
        } catch MLXContinuousBatchCoordinatorError.emptyAdmission {
            #expect(coordinator.isEmpty)
        }
    }

    @Test("realign rejects duplicate and missing rows")
    func realignRejectsDuplicateAndMissingRows() throws {
        var coordinator = MLXContinuousBatchCoordinator<RowState>()
        let first = try coordinator.admit(.init(processor: "a", sampler: "a", cache: "a"))
        let second = try coordinator.admit(.init(processor: "b", sampler: "b", cache: "b"))

        do {
            try coordinator.realign(to: [first, first])
            Issue.record("Expected duplicate row id failure")
        } catch MLXGenerationBatchRowTableError.duplicateRowID(let id) {
            #expect(id == first)
        }

        do {
            try coordinator.realign(to: [first, MLXGenerationBatchRowID(99)])
            Issue.record("Expected missing row id failure")
        } catch MLXGenerationBatchRowTableError.missingRowID(let id) {
            #expect(id == 99)
        }

        #expect(coordinator.orderedRowIDs == [first, second])
    }

    @Test("records row diagnostics for admission update and finish")
    func recordsRowDiagnosticsForLifecycle() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            var coordinator = MLXContinuousBatchCoordinator<RowState>()
            let ids = try coordinator.admitBatch([
                .init(processor: "a", sampler: "a", cache: "a"),
                .init(processor: "b", sampler: "b", cache: "b")
            ])
            try coordinator.updateState(for: ids[0]) { state in
                state.generatedTokenCount = 1
            }
            _ = coordinator.finish(id: ids[1])
        }

        let snapshots: [MLXGenerationBatchRowsSnapshot] = recorded.events.compactMap { event in
            guard case .batchRows(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
        let stages: [MLXGenerationBatchRowsSnapshot.Stage] = [
            .appended,
            .appended,
            .updated,
            .removed
        ]

        #expect(snapshots.map(\.stage) == stages)
        #expect(snapshots.last?.rowIDs == [0])
        #expect(snapshots.last?.affectedRowIDs == [1])
    }

    @Test("shares paged KV prefix blocks between admitted rows")
    func sharesPagedKVPrefixBlocksBetweenAdmittedRows() throws {
        var coordinator = MLXContinuousBatchCoordinator<RowState>(pagedKVBlockCapacity: 3)
        let firstLease = try coordinator.admit(
            .init(processor: "a", sampler: "a", cache: "a"),
            prefixBlockHashes: ["h0", "h1"],
            blockTokenCount: 256
        )
        let secondLease = try coordinator.admit(
            .init(processor: "b", sampler: "b", cache: "b"),
            prefixBlockHashes: ["h0", "h1"],
            blockTokenCount: 256
        )

        #expect(firstLease.blockIDs == secondLease.blockIDs)
        #expect(coordinator.pagedKVRecords.map(\.refCount) == [2, 2])
        _ = coordinator.finish(id: firstLease.rowID)
        #expect(coordinator.pagedKVRecords.map(\.refCount) == [1, 1])
        _ = coordinator.finish(id: secondLease.rowID)
        #expect(coordinator.pagedKVRecords.map(\.refCount) == [0, 0])
    }

    @Test("forks shared paged KV block before row-local mutation")
    func forksSharedPagedKVBlockBeforeRowLocalMutation() throws {
        var coordinator = MLXContinuousBatchCoordinator<RowState>(pagedKVBlockCapacity: 3)
        let firstLease = try coordinator.admit(
            .init(processor: "a", sampler: "a", cache: "a"),
            prefixBlockHashes: ["h0"],
            blockTokenCount: 256
        )
        let secondLease = try coordinator.admit(
            .init(processor: "b", sampler: "b", cache: "b"),
            prefixBlockHashes: ["h0"],
            blockTokenCount: 256
        )

        let forkedID = try coordinator.forkPagedKVBlockForWrite(
            rowID: firstLease.rowID,
            blockID: firstLease.blockIDs[0],
            blockHash: "h0+token",
            tokenCount: 257
        )

        #expect(forkedID != firstLease.blockIDs[0])
        #expect(coordinator.pagedKVBlockIDs(for: firstLease.rowID) == [forkedID])
        #expect(coordinator.pagedKVBlockIDs(for: secondLease.rowID) == secondLease.blockIDs)
        #expect(coordinator.pagedKVRecords.map(\.refCount) == [1, 1])
        #expect(coordinator.pagedKVRecords.map(\.blockHash).sorted() == ["h0", "h0+token"])
    }

    @Test("paged KV admission is transactional when blocks cannot be leased")
    func pagedKVAdmissionIsTransactionalWhenBlocksCannotBeLeased() throws {
        var coordinator = MLXContinuousBatchCoordinator<RowState>(pagedKVBlockCapacity: 1)
        _ = try coordinator.admit(
            .init(processor: "a", sampler: "a", cache: "a"),
            prefixBlockHashes: ["h0"],
            blockTokenCount: 256
        )

        do {
            _ = try coordinator.admit(
                .init(processor: "b", sampler: "b", cache: "b"),
                prefixBlockHashes: ["h1"],
                blockTokenCount: 256
            )
            Issue.record("Expected retained KV block to reject a second prefix")
        } catch MLXPagedKVBlockTableError.noEvictableBlock(let capacity) {
            #expect(capacity == 1)
        }

        #expect(coordinator.count == 1)
        #expect(coordinator.orderedRowIDs == [0])
        #expect(coordinator.pagedKVRecords.map(\.blockHash) == ["h0"])
    }

    private typealias RowState = MLXContinuousBatchRowState<String, String, String>
}
