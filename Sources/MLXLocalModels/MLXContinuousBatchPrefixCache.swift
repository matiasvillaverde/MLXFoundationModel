import Foundation

internal struct MLXContinuousBatchPrefixCache: @unchecked Sendable {
    internal let caches: [KVCache]
    internal let cachedTokenCount: Int
    internal let groupKey: MLXContinuousBatchPrefixCacheGroupKey
    internal let supportsMultiRowMerge: Bool

    internal init(caches: [KVCache], cachedTokenCount: Int) {
        self.caches = caches
        self.cachedTokenCount = cachedTokenCount
        self.supportsMultiRowMerge = Self.supportsMultiRowMerge(caches)
        self.groupKey = Self.groupKey(
            caches: caches,
            cachedTokenCount: cachedTokenCount,
            supportsMultiRowMerge: supportsMultiRowMerge
        )
    }

    private static func groupKey(
        caches: [KVCache],
        cachedTokenCount: Int,
        supportsMultiRowMerge: Bool
    ) -> MLXContinuousBatchPrefixCacheGroupKey {
        guard supportsMultiRowMerge else {
            return .singleton(UUID())
        }
        return .mergeable(
            cachedTokenCount: cachedTokenCount,
            layout: PromptCachePlanner.cacheLayoutFingerprint(for: caches)
        )
    }

    private static func supportsMultiRowMerge(_ caches: [KVCache]) -> Bool {
        !caches.isEmpty && caches.allSatisfy(supportsMultiRowMerge)
    }

    private static func supportsMultiRowMerge(_ cache: KVCache) -> Bool {
        if let cacheList = cache as? CacheList {
            return cacheList.layoutCaches.allSatisfy(supportsMultiRowMerge)
        }
        if cache is ChunkedKVCache {
            return false
        }
        return cache is KVCacheSimple
    }
}
