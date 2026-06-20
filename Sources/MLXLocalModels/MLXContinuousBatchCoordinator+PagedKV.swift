extension MLXContinuousBatchCoordinator {
    @discardableResult
    internal mutating func admit(
        _ state: RowState,
        prefixBlockHashes: [String],
        blockTokenCount: Int
    ) throws -> MLXContinuousBatchPagedKVLease {
        guard pagedKVBlocks != nil else {
            throw MLXContinuousBatchCoordinatorError.pagedKVCacheDisabled
        }

        let id = allocateRowID()
        do {
            let blockIDs = try pagedKVBlocks?.attachPrefix(
                rowID: id,
                blockHashes: prefixBlockHashes,
                tokenCount: blockTokenCount
            ) ?? []
            try rows.append(id: id, payload: state)
            return MLXContinuousBatchPagedKVLease(rowID: id, blockIDs: blockIDs)
        } catch {
            _ = try? pagedKVBlocks?.detach(rowID: id)
            throw error
        }
    }

    @discardableResult
    internal mutating func forkPagedKVBlockForWrite(
        rowID: MLXGenerationBatchRowID,
        blockID: MLXPagedKVBlockID,
        blockHash: String? = nil,
        tokenCount: Int? = nil
    ) throws -> MLXPagedKVBlockID {
        guard pagedKVBlocks != nil else {
            throw MLXContinuousBatchCoordinatorError.pagedKVCacheDisabled
        }
        return try pagedKVBlocks?.forkAttachedBlockForWrite(
            rowID: rowID,
            blockID: blockID,
            blockHash: blockHash,
            tokenCount: tokenCount
        ) ?? blockID
    }

    internal func pagedKVBlockIDs(for rowID: MLXGenerationBatchRowID) -> [MLXPagedKVBlockID] {
        pagedKVBlocks?.attachedBlockIDs(for: rowID) ?? []
    }
}
