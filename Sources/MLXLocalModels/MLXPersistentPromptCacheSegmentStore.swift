import Foundation
import MLX

internal struct MLXPersistentPromptCacheSegment: @unchecked Sendable {
    let blockHash: String
    let blockIndex: Int
    let tokens: [Int]
    let cache: [KVCache]
    let signature: PromptCacheSignature
    let dataURL: URL
}

internal enum MLXPersistentPromptCacheSegmentStore {
    internal typealias Encoder = ([KVCache], [String: String]) throws -> Data
    internal typealias Loader = (URL) throws -> ([KVCache], [String: String])

    private static let directoryName = "IndependentBlocks"
    private static let metadataKey = "mlx.prompt_cache.segment.v1"

    private struct Envelope: Codable, Equatable {
        let blockHash: String
        let blockIndex: Int
        let blockSize: Int
        let tokens: [Int]
        let signature: PromptCacheSignature
    }

    internal static func rootURL() -> URL {
        MLXPersistentPromptCacheBlockStore.rootURL()
            .appendingPathComponent(directoryName, isDirectory: true)
    }

    @discardableResult
    internal static func storeSegments(
        entry: PromptCacheEntry,
        blockSize: Int = 256,
        rootURL: URL = rootURL(),
        now: Date = Date(),
        encoder: Encoder = encode
    ) throws -> [MLXPersistentPromptCacheBlockRecord] {
        let blocks = try makeSegmentBlocks(
            entry: entry,
            blockSize: blockSize,
            encoder: encoder
        )
        let records = try blocks.enumerated().compactMap { index, block in
            if shouldKeepCompactedRotatingBlock(
                block,
                index: index,
                count: blocks.count,
                rootURL: rootURL
            ) {
                return try existingCompactedRotatingBlock(block, rootURL: rootURL)
            }
            return try MLXPersistentPromptCacheBlockStore.storeBlock(
                block,
                rootURL: rootURL,
                now: now
            )
        }
        compactRotatingTipIfNeeded(records: records, rootURL: rootURL, now: now)
        return records
    }

    internal static func makeSegmentBlocks(
        entry: PromptCacheEntry,
        blockSize: Int = 256,
        encoder: Encoder = encode
    ) throws -> [MLXPersistentPromptCachePendingBlock] {
        let segments = PromptCachePlanner.makeBlockEntries(
            from: entry,
            blockSize: blockSize
        )
        return try segments.map { segment in
            try makeSegmentBlock(
                segment,
                blockSize: blockSize,
                encoder: encoder
            )
        }
    }

    internal static func restorePrefixSegments(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        blockSize: Int = 256,
        reusePolicy: PromptCacheReusePolicy = .init(alignment: .exact, prefillStepSize: 1),
        rootURL: URL = rootURL(),
        loader: Loader = load
    ) throws -> [MLXPersistentPromptCacheSegment] {
        guard let hit = try MLXPersistentPromptCacheBlockStore.lookupPrefix(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: blockSize,
            rootURL: rootURL
        ) else {
            recordLookup(
                blockSize: blockSize,
                matchedBlockCount: 0,
                candidateCount: 0,
                selectedIndex: nil,
                reusedTokenCount: 0
            )
            return []
        }

        guard let alignedHit = MLXPersistentPromptCacheBlockStore.cappedToReusePolicy(
            hit,
            requestTokenCount: tokenIds.count,
            reusePolicy: reusePolicy
        ) else {
            recordLookup(hit: hit, reusedTokenCount: 0)
            return []
        }

        let cappedHit = MLXPersistentPromptCacheBlockStore.cappedToHotCacheCapacity(alignedHit)
        guard !cappedHit.dataURLs.isEmpty else {
            recordLookup(hit: alignedHit, reusedTokenCount: 0)
            return []
        }

        var segments: [MLXPersistentPromptCacheSegment] = []
        segments.reserveCapacity(cappedHit.dataURLs.count)

        do {
            for (blockIndex, dataURL) in cappedHit.dataURLs.enumerated() {
                let (cache, metadata) = try loader(dataURL)
                guard let encodedEnvelope = metadata[metadataKey],
                    let envelopeData = Data(base64Encoded: encodedEnvelope)
                else {
                    recordLookup(hit: hit, reusedTokenCount: 0)
                    return []
                }

                let envelope = try JSONDecoder().decode(Envelope.self, from: envelopeData)
                let startIndex = blockIndex * blockSize
                let expectedTokens = Array(tokenIds[startIndex ..< startIndex + blockSize])
                guard envelope.blockIndex == blockIndex,
                    envelope.blockSize == blockSize,
                    envelope.signature == signature,
                    envelope.tokens == expectedTokens
                else {
                    recordLookup(hit: hit, reusedTokenCount: 0)
                    return []
                }

                segments.append(MLXPersistentPromptCacheSegment(
                    blockHash: envelope.blockHash,
                    blockIndex: blockIndex,
                    tokens: envelope.tokens,
                    cache: cache,
                    signature: signature,
                    dataURL: dataURL
                ))
            }
        } catch {
            recordLookup(hit: hit, reusedTokenCount: 0)
            throw error
        }
        recordLookup(
            hit: cappedHit,
            reusedTokenCount: segments.reduce(0) { total, segment in
                total + segment.tokens.count
            }
        )
        return segments
    }

    internal static func restoreBestEntry(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        blockSize: Int = 256,
        reusePolicy: PromptCacheReusePolicy = .init(alignment: .exact, prefillStepSize: 1),
        maxBytes: Int?,
        rootURL: URL = rootURL(),
        loader: Loader = load
    ) throws -> PromptCacheEntry? {
        let segments = try restorePrefixSegments(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: blockSize,
            reusePolicy: reusePolicy,
            rootURL: rootURL,
            loader: loader
        )
        guard !segments.isEmpty else {
            return nil
        }
        return assembleEntry(from: segments, maxBytes: maxBytes)
    }

    internal static func assembleEntry(
        from segments: [MLXPersistentPromptCacheSegment],
        maxBytes: Int?
    ) -> PromptCacheEntry? {
        guard let firstSegment = segments.first else {
            return nil
        }
        guard segments.enumerated().allSatisfy({ offset, segment in
            segment.blockIndex == offset && segment.signature == firstSegment.signature
        }) else {
            return nil
        }

        let tokens = segments.flatMap(\.tokens)
        guard let cache = assembleCache(from: segments, tokenCount: tokens.count) else {
            return nil
        }
        let byteCount = PromptCachePlanner.cacheByteCount(cache)
        guard maxBytes.map({ byteCount <= $0 }) ?? true else {
            return nil
        }

        return PromptCacheEntry(
            tokens: tokens,
            cache: cache,
            signature: firstSegment.signature,
            byteCount: byteCount
        )
    }

    private static func makeSegmentBlock(
        _ segment: PromptCacheBlockEntry,
        blockSize: Int,
        encoder: Encoder
    ) throws -> MLXPersistentPromptCachePendingBlock {
        let envelope = Envelope(
            blockHash: segment.blockHash,
            blockIndex: segment.blockIndex,
            blockSize: blockSize,
            tokens: segment.tokens,
            signature: segment.signature
        )
        let metadata = [
            metadataKey: try JSONEncoder().encode(envelope).base64EncodedString()
        ]
        let payload = try encoder(segment.cache, metadata)
        return MLXPersistentPromptCachePendingBlock(
            blockHash: segment.blockHash,
            blockSize: blockSize,
            tokenCount: segment.tokens.count,
            signature: segment.signature,
            payload: payload,
            payloadKind: .block
        )
    }

    private static func compactRotatingTipIfNeeded(
        records: [MLXPersistentPromptCacheBlockRecord],
        rootURL: URL,
        now: Date
    ) {
        guard records.count >= 2,
            let previous = records.dropLast().last,
            let newest = records.last
        else {
            return
        }
        _ = try? MLXPersistentPromptCacheBlockStore.recordRotatingTipExtension(
            previousTip: descriptor(for: previous, rootURL: rootURL),
            newTip: descriptor(for: newest, rootURL: rootURL),
            now: now
        )
    }

    private static func shouldKeepCompactedRotatingBlock(
        _ block: MLXPersistentPromptCachePendingBlock,
        index: Int,
        count: Int,
        rootURL: URL
    ) -> Bool {
        guard isRotatingCacheSignature(block.signature),
            index < max(count - 2, 0)
        else {
            return false
        }
        return (try? existingCompactedRotatingBlock(block, rootURL: rootURL)) != nil
    }

    private static func existingCompactedRotatingBlock(
        _ block: MLXPersistentPromptCachePendingBlock,
        rootURL: URL
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        try MLXPersistentPromptCacheBlockStore.storedRecord(
            blockHash: block.blockHash,
            signature: block.signature,
            blockSize: block.blockSize,
            rootURL: rootURL,
            payloadKinds: [.compactedRotatingTip]
        )
    }

    private static func descriptor(
        for record: MLXPersistentPromptCacheBlockRecord,
        rootURL: URL
    ) -> MLXPersistentPromptCacheTipDescriptor {
        MLXPersistentPromptCacheTipDescriptor(
            blockHash: record.blockHash,
            blockSize: record.blockSize,
            signature: record.signature,
            payloadKind: .block,
            rootURL: rootURL
        )
    }

    private static func isRotatingCacheSignature(_ signature: PromptCacheSignature) -> Bool {
        signature.cacheLayout?.contains { component in
            component.contains("RotatingKVCache")
        } ?? false
    }

    private static func assembleCache(
        from segments: [MLXPersistentPromptCacheSegment],
        tokenCount: Int
    ) -> [KVCache]? {
        guard let firstCache = segments.first?.cache else {
            return nil
        }
        guard segments.allSatisfy({ $0.cache.count == firstCache.count }) else {
            return nil
        }

        var assembledCache: [KVCache] = []
        assembledCache.reserveCapacity(firstCache.count)
        for cacheIndex in firstCache.indices {
            let cacheParts = segments.map { $0.cache[cacheIndex] }
            guard let cache = assembleCachePart(cacheParts, tokenCount: tokenCount) else {
                return nil
            }
            assembledCache.append(cache)
        }
        return assembledCache
    }

    private static func assembleCachePart(
        _ cacheParts: [KVCache],
        tokenCount: Int
    ) -> KVCache? {
        guard let firstCache = cacheParts.first else {
            return nil
        }
        let states = cacheParts.map(\.state)
        guard let firstState = states.first else {
            return nil
        }
        guard states.allSatisfy({ $0.count == firstState.count }) else {
            return nil
        }

        var assembledCache = firstCache.copy()
        if firstState.isEmpty {
            return setCacheOffset(&assembledCache, tokenCount: tokenCount) ? assembledCache : nil
        }

        var assembledState: [MLXArray] = []
        assembledState.reserveCapacity(firstState.count)
        for stateIndex in firstState.indices {
            let arrays = states.map { $0[stateIndex] }
            guard let concatenatedArray = concatenateStateArrays(arrays) else {
                return nil
            }
            assembledState.append(concatenatedArray)
        }
        assembledCache.state = assembledState
        return setCacheOffset(&assembledCache, tokenCount: tokenCount) ? assembledCache : nil
    }

    private static func concatenateStateArrays(_ arrays: [MLXArray]) -> MLXArray? {
        guard let firstArray = arrays.first,
            firstArray.ndim > 2
        else {
            return nil
        }

        let sequenceAxis = firstArray.ndim - 2
        guard arrays.allSatisfy({ array in
            guard array.ndim == firstArray.ndim else {
                return false
            }
            return (0 ..< firstArray.ndim).allSatisfy { axis in
                axis == sequenceAxis || array.dim(axis) == firstArray.dim(axis)
            }
        }) else {
            return nil
        }
        return concatenated(arrays, axis: sequenceAxis)
    }

    private static func setCacheOffset(
        _ cache: inout KVCache,
        tokenCount: Int
    ) -> Bool {
        if let cacheList = cache as? CacheList {
            return cacheList.layoutCaches.allSatisfy { childCache in
                var childCache = childCache
                return setCacheOffset(&childCache, tokenCount: tokenCount)
            }
        }

        switch cache {
        case let quantizedRotatingCache as QuantizedRotatingKVCache:
            var metadata = quantizedRotatingCache.metaState
            guard metadata.count == 8 else {
                return false
            }
            metadata[3] = String(tokenCount)
            metadata[4] = String(min(tokenCount, Int(metadata[1]) ?? tokenCount))
            quantizedRotatingCache.metaState = metadata
            return quantizedRotatingCache.offset == tokenCount

        case let rotatingCache as RotatingKVCache:
            var metadata = rotatingCache.metaState
            guard metadata.count == 5 else {
                return false
            }
            metadata[3] = String(tokenCount)
            metadata[4] = String(min(tokenCount, Int(metadata[1]) ?? tokenCount))
            rotatingCache.metaState = metadata
            return rotatingCache.offset == tokenCount

        case let quantizedCache as QuantizedKVCache:
            var metadata = quantizedCache.metaState
            guard metadata.count == 4 else {
                return false
            }
            metadata[1] = String(tokenCount)
            quantizedCache.metaState = metadata
            return quantizedCache.offset == tokenCount

        case let chunkedCache as ChunkedKVCache:
            var metadata = chunkedCache.metaState
            guard metadata.count == 2 else {
                return false
            }
            metadata[1] = "0"
            chunkedCache.metaState = metadata
            return chunkedCache.offset == tokenCount

        case is MambaCache:
            return false

        default:
            return cache.offset == tokenCount
        }
    }

    private static func encode(
        cache: [KVCache],
        metadata: [String: String]
    ) throws -> Data {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("segment.safetensors")
        try savePromptCache(url: url, cache: cache, metadata: metadata)
        return try Data(contentsOf: url)
    }

    internal static func load(url: URL) throws -> ([KVCache], [String: String]) {
        let data = try MLXPersistentPromptCacheBlockStore.loadPayload(
            at: url,
            promotionPolicy: .skipIfWouldEvict
        )
        let payload = try loadPromptCache(data: data)
        return (payload.0, payload.1 ?? [:])
    }

    private static func recordLookup(
        hit: MLXPersistentPromptCachePrefixHit,
        reusedTokenCount: Int
    ) {
        recordLookup(
            blockSize: hit.records.first?.blockSize ?? 0,
            matchedBlockCount: hit.matchedBlockCount,
            candidateCount: hit.records.count,
            selectedIndex: hit.records.indices.last,
            reusedTokenCount: reusedTokenCount
        )
    }

    private static func recordLookup(
        blockSize: Int,
        matchedBlockCount: Int,
        candidateCount: Int,
        selectedIndex: Int?,
        reusedTokenCount: Int
    ) {
        MLXGenerationDiagnostics.recordPromptCacheLookup(MLXPromptCacheLookupSnapshot(
            strategy: .persistentSegments,
            blockSize: blockSize,
            matchedBlockCount: matchedBlockCount,
            candidateCount: candidateCount,
            selectedIndex: selectedIndex,
            reusedTokenCount: reusedTokenCount
        ))
    }
}
