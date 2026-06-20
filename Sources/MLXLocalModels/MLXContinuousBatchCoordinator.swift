internal struct MLXContinuousBatchCoordinator<RowState: Sendable>: Sendable {
    internal var rows = MLXGenerationBatchRowTable<RowState>()
    internal var nextRowID = MLXGenerationBatchRowID(0)
    internal var pagedKVBlocks: MLXPagedKVBlockTable?

    internal init(pagedKVBlockCapacity: Int = 0) {
        if pagedKVBlockCapacity > 0 {
            self.pagedKVBlocks = MLXPagedKVBlockTable(capacity: pagedKVBlockCapacity)
        }
    }

    internal var isEmpty: Bool {
        rows.isEmpty
    }

    internal var count: Int {
        rows.count
    }

    internal var orderedRowIDs: [MLXGenerationBatchRowID] {
        rows.orderedIDs
    }

    internal var orderedStates: [RowState] {
        rows.orderedPayloads
    }

    internal var pagedKVRecords: [MLXPagedKVBlockRecord] {
        pagedKVBlocks?.orderedRecords ?? []
    }

    internal subscript(id: MLXGenerationBatchRowID) -> RowState? {
        rows[id]
    }
}
