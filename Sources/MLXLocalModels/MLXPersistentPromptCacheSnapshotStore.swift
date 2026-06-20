import Foundation

internal enum MLXPersistentPromptCacheSnapshotStore {
    internal typealias Encoder = ([KVCache], [String: String]) throws -> Data
    internal typealias Loader = (URL) throws -> ([KVCache], [String: String])

    private static let metadataKey = "mlx.prompt_cache.block_snapshot.v1"

    private struct Envelope: Codable, Equatable {
        let blockHash: String
        let blockSize: Int
        let tokens: [Int]
        let signature: PromptCacheSignature
    }

    @discardableResult
    internal static func storeSnapshot(
        entry: PromptCacheEntry,
        blockSize: Int = 256,
        rootURL: URL = MLXPersistentPromptCacheBlockStore.rootURL(),
        now: Date = Date(),
        encoder: Encoder = encode
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        guard let block = try makeSnapshotBlock(
            entry: entry,
            blockSize: blockSize,
            encoder: encoder
        ) else {
            return nil
        }

        return try MLXPersistentPromptCacheBlockStore.storeBlock(
            block,
            rootURL: rootURL,
            now: now
        )
    }

    internal static func makeSnapshotBlock(
        entry: PromptCacheEntry,
        blockSize: Int = 256,
        encoder: Encoder = encode
    ) throws -> MLXPersistentPromptCachePendingBlock? {
        guard entry.draftCache == nil else {
            return nil
        }

        let blockSize = max(1, blockSize)
        let hashes = PromptCacheBlockIndex.prefixBlockHashes(
            for: entry.tokens,
            blockSize: blockSize
        )
        guard let blockHash = hashes.last else {
            return nil
        }

        let prefixTokenCount = hashes.count * blockSize
        let prefixTokens = Array(entry.tokens.prefix(prefixTokenCount))
        guard let cache = PromptCachePlanner.copyCache(
            entry.cache,
            trimmingToTokenCount: prefixTokenCount
        ) else {
            return nil
        }

        let envelope = Envelope(
            blockHash: blockHash,
            blockSize: blockSize,
            tokens: prefixTokens,
            signature: entry.signature
        )
        let metadata = [
            metadataKey: try JSONEncoder().encode(envelope).base64EncodedString()
        ]
        let payload = try encoder(cache, metadata)
        return MLXPersistentPromptCachePendingBlock(
            blockHash: blockHash,
            blockSize: blockSize,
            tokenCount: prefixTokenCount,
            signature: entry.signature,
            payload: payload,
            payloadKind: .prefixSnapshot
        )
    }

    internal static func restoreBestSnapshot(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        blockSize: Int = 256,
        reusePolicy: PromptCacheReusePolicy = .init(alignment: .exact, prefillStepSize: 1),
        maxBytes: Int?,
        rootURL: URL = MLXPersistentPromptCacheBlockStore.rootURL(),
        now: Date = Date(),
        loader: Loader = load
    ) throws -> PromptCacheEntry? {
        guard let hit = try MLXPersistentPromptCacheBlockStore.lookupBestPrefixSnapshot(
            tokenIds: tokenIds,
            signature: signature,
            blockSize: blockSize,
            rootURL: rootURL,
            now: now
        ) else {
            recordLookup(
                blockSize: blockSize,
                matchedBlockCount: 0,
                candidateCount: 0,
                selectedIndex: nil,
                reusedTokenCount: 0
            )
            return nil
        }

        let reusableTokenCount = reusePolicy.persistentPrefixTokenCount(
            cachedTokenCount: hit.cachedTokenCount,
            requestTokenCount: tokenIds.count,
            blockSize: blockSize
        )
        guard reusableTokenCount > 0 else {
            recordLookup(hit: hit, reusedTokenCount: 0)
            return nil
        }

        let cache: [KVCache]
        let metadata: [String: String]
        do {
            (cache, metadata) = try loader(hit.dataURL)
        } catch {
            recordLookup(hit: hit, reusedTokenCount: 0)
            throw error
        }
        guard let encodedEnvelope = metadata[metadataKey],
            let envelopeData = Data(base64Encoded: encodedEnvelope)
        else {
            recordLookup(hit: hit, reusedTokenCount: 0)
            throw KVCacheError(message: "Persistent prompt cache snapshot metadata missing")
        }

        let envelope = try JSONDecoder().decode(Envelope.self, from: envelopeData)
        guard envelope.blockHash == hit.record.blockHash,
            envelope.blockSize == blockSize,
            envelope.signature == signature,
            envelope.tokens == Array(tokenIds.prefix(hit.cachedTokenCount))
        else {
            recordLookup(hit: hit, reusedTokenCount: 0)
            return nil
        }

        let restoredTokens: [Int]
        if reusableTokenCount < hit.cachedTokenCount {
            let tokensToTrim = hit.cachedTokenCount - reusableTokenCount
            guard canTrimPromptCache(cache),
                trimPromptCache(cache, numTokens: tokensToTrim) == tokensToTrim
            else {
                recordLookup(hit: hit, reusedTokenCount: 0)
                return nil
            }
            restoredTokens = Array(envelope.tokens.prefix(reusableTokenCount))
        } else {
            restoredTokens = envelope.tokens
        }

        let byteCount = PromptCachePlanner.cacheByteCount(cache)
        guard maxBytes.map({ byteCount <= $0 }) ?? true else {
            recordLookup(hit: hit, reusedTokenCount: 0)
            return nil
        }

        recordLookup(hit: hit, reusedTokenCount: reusableTokenCount)
        return PromptCacheEntry(
            tokens: restoredTokens,
            cache: cache,
            signature: signature,
            byteCount: byteCount
        )
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

        let url = directory.appendingPathComponent("cache.safetensors")
        try savePromptCache(url: url, cache: cache, metadata: metadata)
        return try Data(contentsOf: url)
    }

    internal static func load(url: URL) throws -> ([KVCache], [String: String]) {
        let data = try MLXPersistentPromptCacheBlockStore.loadPayload(at: url)
        let payload = try loadPromptCache(data: data)
        return (payload.0, payload.1 ?? [:])
    }

    private static func recordLookup(
        hit: MLXPersistentPromptCacheSnapshotHit,
        reusedTokenCount: Int
    ) {
        recordLookup(
            blockSize: hit.record.blockSize,
            matchedBlockCount: hit.matchedBlockCount,
            candidateCount: 1,
            selectedIndex: hit.matchedBlockCount - 1,
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
            strategy: .persistentSnapshot,
            blockSize: blockSize,
            matchedBlockCount: matchedBlockCount,
            candidateCount: candidateCount,
            selectedIndex: selectedIndex,
            reusedTokenCount: reusedTokenCount
        ))
    }
}
