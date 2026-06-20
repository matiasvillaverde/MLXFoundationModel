import Foundation

internal enum MLXPromptCacheEntryStore {
    private struct PendingPersistentBlock {
        let block: MLXPersistentPromptCachePendingBlock
        let rootURL: URL
    }

    private struct StoredPersistentBlocks {
        let hashes: Set<String>
        let segmentRecords: [MLXPersistentPromptCacheBlockRecord]
    }

    internal struct Request {
        let parameters: GenerateParameters
        let cacheVariant: String?
        let promptCacheIdentity: PromptCacheIdentity?
        let maxBytes: Int?
        let reusePromptCache: Bool
        let runtimePreferences: ModelRuntimePreferences
    }

    internal static func update(
        _ entries: inout [PromptCacheEntry],
        tokenIDs: [Int],
        reusableState: PromptCacheReusableState,
        request: Request
    ) {
        guard request.reusePromptCache else {
            entries.removeAll()
            return
        }
        guard let entry = makeEntry(tokenIDs, reusableState: reusableState, request: request) else {
            return
        }

        PromptCachePlanner.store(entry, in: &entries, maxBytes: request.maxBytes)
        persistIfNeeded(entry, request: request)
    }

    private static func makeEntry(
        _ tokenIDs: [Int],
        reusableState: PromptCacheReusableState,
        request: Request
    ) -> PromptCacheEntry? {
        PromptCachePlanner.makeEntry(
            tokenIds: tokenIDs,
            cache: reusableState.cache,
            draftCache: reusableState.draftCache,
            parameters: request.parameters,
            cacheVariant: request.cacheVariant,
            promptCacheIdentity: request.promptCacheIdentity,
            maxBytes: request.maxBytes
        )
    }

    private static func persistIfNeeded(
        _ entry: PromptCacheEntry,
        request: Request
    ) {
        guard request.runtimePreferences.promptCachePolicy == .persistent,
            entry.draftCache == nil
        else {
            return
        }

        invalidateStalePersistentPromptCacheSignatures(for: entry)
        let protectedStorageHashes = storePersistentPromptCacheBlocks(
            for: entry,
            request: request
        )
        enforcePersistentPromptCacheBudgets(
            request: request,
            protectedStorageHashes: protectedStorageHashes
        )
    }

    private static func storePersistentPromptCacheBlocks(
        for entry: PromptCacheEntry,
        request: Request
    ) -> Set<String> {
        let blocks = pendingPersistentPromptCacheBlocks(for: entry)
        enforcePersistentPromptCacheAdmission(
            request: request,
            incomingByteCount: blocks.reduce(0) { total, item in
                total + item.block.byteCount
            }
        )

        let now = Date()
        let stored = storePersistentPromptCacheBlockRecords(
            blocks,
            entry: entry,
            now: now
        )
        compactRotatingTipIfNeeded(records: stored.segmentRecords, signature: entry.signature, now: now)
        return stored.hashes
    }

    private static func storePersistentPromptCacheBlockRecords(
        _ blocks: [PendingPersistentBlock],
        entry: PromptCacheEntry,
        now: Date
    ) -> StoredPersistentBlocks {
        var hashes: Set<String> = []
        var segmentRecords: [MLXPersistentPromptCacheBlockRecord] = []
        var segmentIndex = 0
        let segmentCount = blocks.count { isSegmentBlock($0) }

        for item in blocks {
            let currentSegmentIndex = isSegmentBlock(item) ? segmentIndex : nil
            if currentSegmentIndex != nil {
                segmentIndex += 1
            }
            if let record = storePersistentPromptCacheBlock(
                item,
                entry: entry,
                segmentIndex: currentSegmentIndex,
                segmentCount: segmentCount,
                now: now
            ) {
                hashes.insert(record.storageHash)
                if currentSegmentIndex != nil {
                    segmentRecords.append(record)
                }
            }
        }
        return StoredPersistentBlocks(hashes: hashes, segmentRecords: segmentRecords)
    }

    private static func storePersistentPromptCacheBlock(
        _ item: PendingPersistentBlock,
        entry: PromptCacheEntry,
        segmentIndex: Int?,
        segmentCount: Int,
        now: Date
    ) -> MLXPersistentPromptCacheBlockRecord? {
        if shouldKeepCompactedRotatingSegment(
            item,
            entry: entry,
            segmentIndex: segmentIndex,
            segmentCount: segmentCount
        ) {
            return try? existingCompactedRotatingSegmentRecord(item)
        }
        return try? MLXPersistentPromptCacheBlockStore.storeBlock(
            item.block,
            rootURL: item.rootURL,
            now: now
        )
    }

    private static func pendingPersistentPromptCacheBlocks(
        for entry: PromptCacheEntry
    ) -> [PendingPersistentBlock] {
        var blocks: [PendingPersistentBlock] = []
        if let snapshot = try? MLXPersistentPromptCacheSnapshotStore.makeSnapshotBlock(entry: entry) {
            blocks.append(PendingPersistentBlock(
                block: snapshot,
                rootURL: MLXPersistentPromptCacheBlockStore.rootURL()
            ))
        }
        if let segments = try? MLXPersistentPromptCacheSegmentStore.makeSegmentBlocks(entry: entry) {
            blocks.append(contentsOf: segments.map { segment in
                PendingPersistentBlock(
                    block: segment,
                    rootURL: MLXPersistentPromptCacheSegmentStore.rootURL()
                )
            })
        }
        return blocks
    }

    private static func shouldKeepCompactedRotatingSegment(
        _ item: PendingPersistentBlock,
        entry: PromptCacheEntry,
        segmentIndex: Int?,
        segmentCount: Int
    ) -> Bool {
        guard let segmentIndex,
            isRotatingCacheSignature(entry.signature),
            segmentIndex < max(segmentCount - 2, 0)
        else {
            return false
        }
        return (try? existingCompactedRotatingSegmentRecord(item)) != nil
    }

    private static func existingCompactedRotatingSegmentRecord(
        _ item: PendingPersistentBlock
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        try MLXPersistentPromptCacheBlockStore.storedRecord(
            blockHash: item.block.blockHash,
            signature: item.block.signature,
            blockSize: item.block.blockSize,
            rootURL: item.rootURL,
            payloadKinds: [.compactedRotatingTip]
        )
    }

    private static func compactRotatingTipIfNeeded(
        records: [MLXPersistentPromptCacheBlockRecord],
        signature: PromptCacheSignature,
        now: Date
    ) {
        guard isRotatingCacheSignature(signature),
            records.count >= 2,
            let previous = records.dropLast().last,
            let newest = records.last
        else {
            return
        }
        _ = try? MLXPersistentPromptCacheBlockStore.recordRotatingTipExtension(
            previousTip: descriptor(for: previous),
            newTip: descriptor(for: newest),
            now: now
        )
    }

    private static func descriptor(
        for record: MLXPersistentPromptCacheBlockRecord
    ) -> MLXPersistentPromptCacheTipDescriptor {
        MLXPersistentPromptCacheTipDescriptor(
            blockHash: record.blockHash,
            blockSize: record.blockSize,
            signature: record.signature,
            payloadKind: .block,
            rootURL: MLXPersistentPromptCacheSegmentStore.rootURL()
        )
    }

    private static func isSegmentBlock(_ item: PendingPersistentBlock) -> Bool {
        item.block.payloadKind == .block &&
            item.rootURL == MLXPersistentPromptCacheSegmentStore.rootURL()
    }

    private static func isRotatingCacheSignature(_ signature: PromptCacheSignature) -> Bool {
        signature.cacheLayout?.contains { component in
            component.contains("RotatingKVCache")
        } ?? false
    }

    private static func invalidateStalePersistentPromptCacheSignatures(
        for entry: PromptCacheEntry
    ) {
        _ = try? MLXPersistentPromptCacheBlockStore.invalidateStaleSignatures(
            expectedSignature: entry.signature
        )
        _ = try? MLXPersistentPromptCacheBlockStore.invalidateStaleSignatures(
            expectedSignature: entry.signature,
            rootURL: MLXPersistentPromptCacheSegmentStore.rootURL()
        )
    }

    private static func enforcePersistentPromptCacheAdmission(
        request: Request,
        incomingByteCount: Int
    ) {
        try? MLXPersistentPromptCacheBudgetEnforcer.enforceAllBeforeInsert(
            limitBytes: request.runtimePreferences.persistentPromptCacheTotalByteLimit,
            incomingByteCount: incomingByteCount
        )
    }

    private static func enforcePersistentPromptCacheBudgets(
        request: Request,
        protectedStorageHashes: Set<String>
    ) {
        try? MLXPersistentPromptCacheBudgetEnforcer.enforceAll(
            limitBytes: request.runtimePreferences.persistentPromptCacheTotalByteLimit,
            protectedStorageHashes: protectedStorageHashes
        )
    }
}
