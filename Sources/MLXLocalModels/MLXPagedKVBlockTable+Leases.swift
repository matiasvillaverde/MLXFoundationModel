extension MLXPagedKVBlockTable {
    @discardableResult
    internal mutating func attachPrefix(
        rowID: MLXGenerationBatchRowID,
        blockHashes: [String],
        tokenCount: Int
    ) throws -> [MLXPagedKVBlockID] {
        let blockIDs = try blockHashes.map { blockHash in
            try reusableBlockID(blockHash: blockHash, tokenCount: tokenCount)
        }
        try attachPrefix(rowID: rowID, blockIDs: blockIDs)
        return blockIDs
    }

    internal mutating func attachPrefix(
        rowID: MLXGenerationBatchRowID,
        blockIDs: [MLXPagedKVBlockID]
    ) throws {
        try validateUnique(blockIDs)
        try blockIDs.forEach { _ = try requireRecord($0) }
        _ = try detach(rowID: rowID, recordsDiagnostics: false)
        for id in blockIDs {
            try retain(id, recordsDiagnostics: false)
        }
        blockIDsByRowID[rowID] = blockIDs
        record(stage: .attached, affectedIDs: blockIDs, rowID: rowID)
    }

    @discardableResult
    internal mutating func forkAttachedBlockForWrite(
        rowID: MLXGenerationBatchRowID,
        blockID: MLXPagedKVBlockID,
        blockHash: String? = nil,
        tokenCount: Int? = nil
    ) throws -> MLXPagedKVBlockID {
        guard let rowBlockIDs = blockIDsByRowID[rowID],
            let blockIndex = rowBlockIDs.firstIndex(of: blockID)
        else {
            throw MLXPagedKVBlockTableError.missingRowBlock(
                rowID: rowID,
                blockID: blockID
            )
        }

        let forkedID = try forkForWrite(
            blockID,
            blockHash: blockHash,
            tokenCount: tokenCount
        )
        blockIDsByRowID[rowID]?[blockIndex] = forkedID
        if forkedID != blockID {
            record(stage: .attached, affectedIDs: [forkedID], rowID: rowID)
        }
        return forkedID
    }

    @discardableResult
    internal mutating func detach(rowID: MLXGenerationBatchRowID) throws -> [MLXPagedKVBlockID] {
        try detach(rowID: rowID, recordsDiagnostics: true)
    }

    internal mutating func removeAll(keepingCapacity: Bool = true) {
        recordsByID.removeAll(keepingCapacity: keepingCapacity)
        idsByHash.removeAll(keepingCapacity: keepingCapacity)
        blockIDsByRowID.removeAll(keepingCapacity: keepingCapacity)
        freeIDs = Self.makeFreeIDs(capacity: capacity)
        record(stage: .cleared, affectedIDs: [])
    }

    internal mutating func detach(
        rowID: MLXGenerationBatchRowID,
        recordsDiagnostics: Bool
    ) throws -> [MLXPagedKVBlockID] {
        guard let blockIDs = blockIDsByRowID.removeValue(forKey: rowID) else {
            return []
        }
        for id in blockIDs {
            try release(id, recordsDiagnostics: false)
        }
        if recordsDiagnostics {
            record(stage: .detached, affectedIDs: blockIDs, rowID: rowID)
        }
        return blockIDs
    }
}
