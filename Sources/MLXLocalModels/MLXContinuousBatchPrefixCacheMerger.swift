import MLX

internal enum MLXContinuousBatchPrefixCacheMerger {
    internal static func initialCache(
        model: any LanguageModel,
        parameters: GenerateParameters,
        requests: [MLXContinuousBatchPrefillRequest]
    ) throws -> [KVCache] {
        guard let prefixCache = requests[0].prefixCache else {
            return model.newCache(parameters: parameters)
        }
        try validatePrefixCachePresence(in: requests)

        if prefixCache.supportsMultiRowMerge {
            return try merge(cacheRows: requests.map { $0.prefixCache?.caches ?? [] })
        }
        guard requests.count == 1 else {
            throw MLXContinuousBatchPrefillError.unsupportedPrefixCacheType(
                typeName: typeName(for: prefixCache.caches)
            )
        }
        return prefixCache.caches
    }

    private static func validatePrefixCachePresence(
        in requests: [MLXContinuousBatchPrefillRequest]
    ) throws {
        for index in requests.indices where requests[index].prefixCache == nil {
            throw MLXContinuousBatchPrefillError.incompatiblePrefixCache(rowIndex: index)
        }
    }

    private static func merge(cacheRows: [[KVCache]]) throws -> [KVCache] {
        let firstLayerCount = cacheRows[0].count
        try validateLayerCounts(cacheRows, expected: firstLayerCount)
        return try (0 ..< firstLayerCount).map { layerIndex in
            try mergeLayer(cacheRows.map { $0[layerIndex] }, layerIndex: layerIndex)
        }
    }

    private static func validateLayerCounts(
        _ cacheRows: [[KVCache]],
        expected: Int
    ) throws {
        for rowIndex in cacheRows.indices where cacheRows[rowIndex].count != expected {
            throw MLXContinuousBatchPrefillError.mismatchedPrefixCacheLayerCount(
                rowIndex: rowIndex
            )
        }
    }

    private static func mergeLayer(
        _ caches: [KVCache],
        layerIndex: Int
    ) throws -> KVCache {
        guard let first = caches.first else {
            throw MLXContinuousBatchPrefillError.invalidPrefixCacheState(
                rowIndex: 0,
                layerIndex: layerIndex
            )
        }
        if first is CacheList {
            return try mergeCacheLists(caches, layerIndex: layerIndex)
        }
        if first is ChunkedKVCache {
            throw unsupported(first)
        }
        if first is KVCacheSimple {
            return try mergeSimpleCaches(caches, layerIndex: layerIndex)
        }
        throw unsupported(first)
    }

    private static func mergeCacheLists(
        _ caches: [KVCache],
        layerIndex: Int
    ) throws -> KVCache {
        let cacheLists = try cacheLists(caches, layerIndex: layerIndex)
        let childCount = cacheLists[0].layoutCaches.count
        try validateCacheListChildCounts(cacheLists, expected: childCount, layerIndex: layerIndex)
        let children = try (0 ..< childCount).map { childIndex in
            try mergeLayer(cacheLists.map { $0.layoutCaches[childIndex] }, layerIndex: layerIndex)
        }
        return CacheList(caches: children)
    }

    private static func cacheLists(
        _ caches: [KVCache],
        layerIndex: Int
    ) throws -> [CacheList] {
        try caches.enumerated().map { rowIndex, cache in
            guard let cacheList = cache as? CacheList else {
                throw MLXContinuousBatchPrefillError.incompatiblePrefixCache(rowIndex: rowIndex)
            }
            guard !cacheList.layoutCaches.isEmpty else {
                throw MLXContinuousBatchPrefillError.invalidPrefixCacheState(
                    rowIndex: rowIndex,
                    layerIndex: layerIndex
                )
            }
            return cacheList
        }
    }

    private static func validateCacheListChildCounts(
        _ cacheLists: [CacheList],
        expected: Int,
        layerIndex: Int
    ) throws {
        for rowIndex in cacheLists.indices where cacheLists[rowIndex].layoutCaches.count != expected {
            throw MLXContinuousBatchPrefillError.invalidPrefixCacheState(
                rowIndex: rowIndex,
                layerIndex: layerIndex
            )
        }
    }

    private static func mergeSimpleCaches(
        _ caches: [KVCache],
        layerIndex: Int
    ) throws -> KVCache {
        let states = try simpleStates(caches, layerIndex: layerIndex)
        try validateSimpleStates(states, layerIndex: layerIndex)
        let merged = KVCacheSimple()
        let keys = concatenated(states.map(\.keys), axis: 0)
        let values = concatenated(states.map(\.values), axis: 0)
        eval(keys, values)
        merged.state = [keys, values]
        return merged
    }

    private static func simpleStates(
        _ caches: [KVCache],
        layerIndex: Int
    ) throws -> [SimpleState] {
        try caches.enumerated().map { rowIndex, cache in
            guard typeName(for: cache) == typeName(for: caches[0]) else {
                throw MLXContinuousBatchPrefillError.incompatiblePrefixCache(rowIndex: rowIndex)
            }
            let state = cache.state
            guard state.count == 2 else {
                throw MLXContinuousBatchPrefillError.invalidPrefixCacheState(
                    rowIndex: rowIndex,
                    layerIndex: layerIndex
                )
            }
            return SimpleState(rowIndex: rowIndex, offset: cache.offset, keys: state[0], values: state[1])
        }
    }

    private static func validateSimpleStates(
        _ states: [SimpleState],
        layerIndex: Int
    ) throws {
        let reference = states[0]
        for state in states {
            try validate(state.keys, reference: reference.keys, state: state, layerIndex: layerIndex)
            try validate(state.values, reference: reference.values, state: state, layerIndex: layerIndex)
            guard state.offset == reference.offset else {
                throw MLXContinuousBatchPrefillError.mismatchedPrefixCacheOffset(
                    rowIndex: state.rowIndex,
                    layerIndex: layerIndex
                )
            }
        }
    }

    private static func validate(
        _ array: MLXArray,
        reference: MLXArray,
        state: SimpleState,
        layerIndex: Int
    ) throws {
        guard array.ndim == reference.ndim, array.ndim > 0, array.dim(0) == 1 else {
            throw MLXContinuousBatchPrefillError.invalidPrefixCacheState(
                rowIndex: state.rowIndex,
                layerIndex: layerIndex
            )
        }
        guard shapesMatch(array.shape, reference.shape) else {
            throw MLXContinuousBatchPrefillError.mismatchedPrefixCacheShape(
                rowIndex: state.rowIndex,
                layerIndex: layerIndex
            )
        }
    }

    private static func shapesMatch(_ shape: [Int], _ reference: [Int]) -> Bool {
        shape.count == reference.count
            && zip(shape.dropFirst(), reference.dropFirst()).allSatisfy(==)
    }

    private static func unsupported(_ cache: KVCache) -> MLXContinuousBatchPrefillError {
        .unsupportedPrefixCacheType(typeName: typeName(for: cache))
    }

    private static func typeName(for cache: KVCache) -> String {
        String(describing: type(of: cache))
    }

    private static func typeName(for caches: [KVCache]) -> String {
        caches.first.map(typeName(for:)) ?? "empty"
    }

    private struct SimpleState {
        let rowIndex: Int
        let offset: Int
        let keys: MLXArray
        let values: MLXArray
    }
}
