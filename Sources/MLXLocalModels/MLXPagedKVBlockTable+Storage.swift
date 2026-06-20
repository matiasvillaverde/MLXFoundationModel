extension MLXPagedKVBlockTable {
    internal mutating func reserveBlockID() throws -> MLXPagedKVBlockID {
        if let id = freeIDs.popLast() {
            return id
        }
        guard let evictedRecord = leastRecentlyUsedEvictableRecord() else {
            throw MLXPagedKVBlockTableError.noEvictableBlock(capacity: capacity)
        }
        removeRecord(evictedRecord)
        record(stage: .evicted, affectedIDs: [evictedRecord.id])
        return evictedRecord.id
    }

    internal mutating func reusableBlockID(
        blockHash: String,
        tokenCount: Int
    ) throws -> MLXPagedKVBlockID {
        if let blockID = blockIDs(matchingHash: blockHash).first {
            try touch(blockID)
            return blockID
        }

        let blockID = try allocate(blockHash: blockHash, tokenCount: tokenCount)
        try release(blockID, recordsDiagnostics: false)
        return blockID
    }

    internal mutating func insertRecord(
        id: MLXPagedKVBlockID,
        blockHash: String,
        tokenCount: Int,
        refCount: Int
    ) {
        let record = MLXPagedKVBlockRecord(
            id: id,
            blockHash: blockHash,
            tokenCount: max(0, tokenCount),
            refCount: max(0, refCount),
            lastAccessTick: nextAccessTick()
        )
        recordsByID[id] = record
        idsByHash[blockHash, default: []].insert(id)
    }

    internal mutating func removeRecord(_ record: MLXPagedKVBlockRecord) {
        recordsByID.removeValue(forKey: record.id)
        removeHashIndex(record)
    }

    internal mutating func removeHashIndex(_ record: MLXPagedKVBlockRecord) {
        idsByHash[record.blockHash]?.remove(record.id)
        if idsByHash[record.blockHash]?.isEmpty == true {
            idsByHash.removeValue(forKey: record.blockHash)
        }
    }
}
