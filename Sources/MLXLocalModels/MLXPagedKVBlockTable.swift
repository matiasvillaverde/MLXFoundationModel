internal struct MLXPagedKVBlockTable: Sendable {
    internal let capacity: Int
    internal var nextTick: UInt64 = 0
    internal var freeIDs: [MLXPagedKVBlockID]
    internal var recordsByID: [MLXPagedKVBlockID: MLXPagedKVBlockRecord] = [:]
    internal var idsByHash: [String: Set<MLXPagedKVBlockID>] = [:]
    internal var blockIDsByRowID: [MLXGenerationBatchRowID: [MLXPagedKVBlockID]] = [:]

    internal init(capacity: Int) {
        let boundedCapacity = max(0, capacity)
        self.capacity = boundedCapacity
        self.freeIDs = Self.makeFreeIDs(capacity: boundedCapacity)
    }

    internal var count: Int {
        recordsByID.count
    }

    internal var freeCount: Int {
        freeIDs.count
    }

    internal var evictableCount: Int {
        recordsByID.values.filter(\.isEvictable).count
    }

    internal var orderedRecords: [MLXPagedKVBlockRecord] {
        recordsByID.values.sorted { $0.id < $1.id }
    }

    internal func record(for id: MLXPagedKVBlockID) -> MLXPagedKVBlockRecord? {
        recordsByID[id]
    }

    internal func blockIDs(matchingHash blockHash: String) -> [MLXPagedKVBlockID] {
        (idsByHash[blockHash] ?? []).sorted()
    }

    internal func attachedBlockIDs(for rowID: MLXGenerationBatchRowID) -> [MLXPagedKVBlockID] {
        blockIDsByRowID[rowID] ?? []
    }
}
