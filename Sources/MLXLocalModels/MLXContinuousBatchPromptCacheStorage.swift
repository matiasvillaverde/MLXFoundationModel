import MLX

internal struct MLXContinuousBatchPromptCacheStorage: @unchecked Sendable {
    internal let tokenIDs: [Int]
    internal let request: MLXPromptCacheEntryStore.Request

    internal func store(
        cache: [KVCache],
        rowIndex: Int,
        rowCount: Int,
        entries: inout [PromptCacheEntry]
    ) throws {
        let rowCache = try cacheRow(cache, rowIndex: rowIndex, rowCount: rowCount)
        MLXPromptCacheEntryStore.update(
            &entries,
            tokenIDs: tokenIDs,
            reusableState: PromptCacheReusableState(cache: rowCache),
            request: request
        )
    }

    private func cacheRow(
        _ cache: [KVCache],
        rowIndex: Int,
        rowCount: Int
    ) throws -> [KVCache] {
        guard rowIndex < rowCount else {
            throw MLXContinuousBatchPrefillError.incompatiblePrefixCache(rowIndex: rowIndex)
        }
        let indexArray = MLXArray([rowIndex])
        return cache.map { cache in
            Self.filtered(cache, indexArray: indexArray, rowCount: rowCount)
        }
    }

    private static func filtered(
        _ cache: KVCache,
        indexArray: MLXArray,
        rowCount: Int
    ) -> KVCache {
        var copy = cache.copy()
        let state = copy.state
        var didFilter = false
        let filteredState = state.map { array in
            guard array.ndim > 0, array.dim(0) == rowCount else {
                return array
            }
            didFilter = true
            return array[indexArray]
        }
        if didFilter {
            copy.state = filteredState
        }
        return copy
    }
}
