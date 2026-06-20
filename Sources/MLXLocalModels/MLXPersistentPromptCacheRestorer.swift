import Foundation

internal enum MLXPersistentPromptCacheRestorer {
    internal static func restoreBestEntry(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        blockSize: Int = 256,
        reusePolicy: PromptCacheReusePolicy = .init(alignment: .exact, prefillStepSize: 1),
        maxBytes: Int?,
        blockRootURL: URL = MLXPersistentPromptCacheBlockStore.rootURL(),
        segmentRootURL: URL = MLXPersistentPromptCacheSegmentStore.rootURL(),
        segmentLoader: MLXPersistentPromptCacheSegmentStore.Loader =
            MLXPersistentPromptCacheSegmentStore.load,
        snapshotLoader: MLXPersistentPromptCacheSnapshotStore.Loader =
            MLXPersistentPromptCacheSnapshotStore.load
    ) -> PromptCacheEntry? {
        let segmentEntry = try? MLXPersistentPromptCacheSegmentStore.restoreBestEntry(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: blockSize,
            reusePolicy: reusePolicy,
            maxBytes: maxBytes,
            rootURL: segmentRootURL,
            loader: segmentLoader
        )
        let snapshotEntry = try? MLXPersistentPromptCacheSnapshotStore.restoreBestSnapshot(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: blockSize,
            reusePolicy: reusePolicy,
            maxBytes: maxBytes,
            rootURL: blockRootURL,
            loader: snapshotLoader
        )
        return deepestEntry([segmentEntry, snapshotEntry].compactMap { $0 })
    }

    private static func deepestEntry(_ entries: [PromptCacheEntry]) -> PromptCacheEntry? {
        entries.max { lhs, rhs in
            if lhs.tokens.count == rhs.tokens.count {
                return lhs.byteCount > rhs.byteCount
            }
            return lhs.tokens.count < rhs.tokens.count
        }
    }
}
