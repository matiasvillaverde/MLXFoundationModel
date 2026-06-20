extension MLXPagedKVBlockTable {
    internal func requireRecord(
        _ id: MLXPagedKVBlockID
    ) throws -> MLXPagedKVBlockRecord {
        guard let record = recordsByID[id] else {
            throw MLXPagedKVBlockTableError.missingBlockID(id)
        }
        return record
    }

    internal func leastRecentlyUsedEvictableRecord() -> MLXPagedKVBlockRecord? {
        recordsByID.values
            .filter(\.isEvictable)
            .min { lhs, rhs in
                if lhs.lastAccessTick == rhs.lastAccessTick {
                    return lhs.id < rhs.id
                }
                return lhs.lastAccessTick < rhs.lastAccessTick
            }
    }

    internal func validateUnique(_ blockIDs: [MLXPagedKVBlockID]) throws {
        var seenIDs: Set<MLXPagedKVBlockID> = []
        for id in blockIDs {
            guard seenIDs.insert(id).inserted else {
                throw MLXPagedKVBlockTableError.duplicateBlockID(id)
            }
        }
    }

    internal mutating func nextAccessTick() -> UInt64 {
        nextTick &+= 1
        return nextTick
    }

    internal static func makeFreeIDs(capacity: Int) -> [MLXPagedKVBlockID] {
        var blockIDs: [MLXPagedKVBlockID] = []
        blockIDs.reserveCapacity(capacity)
        for rawValue in 0 ..< capacity {
            blockIDs.append(MLXPagedKVBlockID(rawValue))
        }
        blockIDs.reverse()
        return blockIDs
    }

    internal func record(
        stage: MLXPagedKVBlockTableSnapshot.Stage,
        affectedIDs: [MLXPagedKVBlockID],
        rowID: MLXGenerationBatchRowID? = nil
    ) {
        let records = orderedRecords
        MLXGenerationDiagnostics.recordPagedKVBlocks(.init(
            stage: stage,
            capacity: capacity,
            usedCount: records.count,
            freeCount: freeIDs.count,
            evictableCount: records.filter(\.isEvictable).count,
            blockIDs: records.map(\.id.rawValue),
            affectedBlockIDs: affectedIDs.map(\.rawValue).sorted(),
            refCounts: records.map(\.refCount),
            rowID: rowID?.rawValue
        ))
    }
}
