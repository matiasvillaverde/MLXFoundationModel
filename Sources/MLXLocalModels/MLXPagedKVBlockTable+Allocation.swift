extension MLXPagedKVBlockTable {
    @discardableResult
    internal mutating func allocate(
        blockHash: String,
        tokenCount: Int
    ) throws -> MLXPagedKVBlockID {
        let reservedID = try reserveBlockID()
        insertRecord(
            id: reservedID,
            blockHash: blockHash,
            tokenCount: tokenCount,
            refCount: 1
        )
        record(stage: .allocated, affectedIDs: [reservedID])
        return reservedID
    }

    internal mutating func retain(_ id: MLXPagedKVBlockID) throws {
        try retain(id, recordsDiagnostics: true)
    }

    internal mutating func release(_ id: MLXPagedKVBlockID) throws {
        try release(id, recordsDiagnostics: true)
    }

    @discardableResult
    internal mutating func forkForWrite(
        _ id: MLXPagedKVBlockID,
        blockHash: String? = nil,
        tokenCount: Int? = nil
    ) throws -> MLXPagedKVBlockID {
        let record = try requireRecord(id)
        guard record.refCount > 1 else {
            try update(id, blockHash: blockHash, tokenCount: tokenCount)
            return id
        }

        let reservedID = try reserveBlockID()
        var originalRecord = record
        originalRecord.refCount -= 1
        originalRecord.lastAccessTick = nextAccessTick()
        recordsByID[id] = originalRecord
        insertRecord(
            id: reservedID,
            blockHash: blockHash ?? record.blockHash,
            tokenCount: tokenCount ?? record.tokenCount,
            refCount: 1
        )
        self.record(stage: .forked, affectedIDs: [id, reservedID])
        return reservedID
    }

    internal mutating func update(
        _ id: MLXPagedKVBlockID,
        blockHash: String? = nil,
        tokenCount: Int? = nil
    ) throws {
        var record = try requireRecord(id)
        if let blockHash, blockHash != record.blockHash {
            removeHashIndex(record)
            record.blockHash = blockHash
        }
        if let tokenCount {
            record.tokenCount = tokenCount
        }
        record.lastAccessTick = nextAccessTick()
        recordsByID[id] = record
        idsByHash[record.blockHash, default: []].insert(id)
        self.record(stage: .updated, affectedIDs: [id])
    }

    internal mutating func touch(_ id: MLXPagedKVBlockID) throws {
        var record = try requireRecord(id)
        record.lastAccessTick = nextAccessTick()
        recordsByID[id] = record
        self.record(stage: .touched, affectedIDs: [id])
    }

    internal mutating func retain(
        _ id: MLXPagedKVBlockID,
        recordsDiagnostics: Bool
    ) throws {
        var record = try requireRecord(id)
        record.refCount += 1
        record.lastAccessTick = nextAccessTick()
        recordsByID[id] = record
        if recordsDiagnostics {
            self.record(stage: .retained, affectedIDs: [id])
        }
    }

    internal mutating func release(
        _ id: MLXPagedKVBlockID,
        recordsDiagnostics: Bool
    ) throws {
        var record = try requireRecord(id)
        guard record.refCount > 0 else {
            throw MLXPagedKVBlockTableError.releaseUnderflow(id)
        }
        record.refCount -= 1
        record.lastAccessTick = nextAccessTick()
        recordsByID[id] = record
        if recordsDiagnostics {
            self.record(stage: .released, affectedIDs: [id])
        }
    }
}
