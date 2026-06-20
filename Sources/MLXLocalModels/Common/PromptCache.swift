import Foundation
import CryptoKit
import MLX

internal struct PromptCacheSignature: Codable, Equatable, Sendable {
    let cacheVariant: String?
    let cacheLayout: [String]?
    let maxKVSize: Int?
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let indexCacheFrequency: Int?
    let prefillStepSize: Int
    let promptCacheIdentity: PromptCacheIdentity?

    init(
        parameters: GenerateParameters,
        cacheVariant: String? = nil,
        cacheLayout: [String]? = nil,
        promptCacheIdentity: PromptCacheIdentity? = nil
    ) {
        self.cacheVariant = cacheVariant
        self.cacheLayout = cacheLayout
        self.maxKVSize = parameters.maxKVSize
        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.indexCacheFrequency = parameters.indexCacheFrequency
        self.prefillStepSize = parameters.prefillStepSize
        self.promptCacheIdentity = promptCacheIdentity
    }

    internal func persistentCacheInvalidationScopeMatches(
        _ other: PromptCacheSignature
    ) -> Bool {
        guard promptCacheIdentity != nil else {
            return false
        }
        return promptCacheIdentity == other.promptCacheIdentity &&
            cacheVariant == other.cacheVariant &&
            prefillStepSize == other.prefillStepSize
    }
}

internal struct PromptCacheEntry: @unchecked Sendable {
    let tokens: [Int]
    let cache: [KVCache]
    let draftCache: [KVCache]?
    let signature: PromptCacheSignature
    let byteCount: Int
    var activeLeaseIDs: Set<UUID>

    internal init(
        tokens: [Int],
        cache: [KVCache],
        draftCache: [KVCache]? = nil,
        signature: PromptCacheSignature,
        byteCount: Int,
        activeLeaseIDs: Set<UUID> = []
    ) {
        self.tokens = tokens
        self.cache = cache
        self.draftCache = draftCache
        self.signature = signature
        self.byteCount = byteCount
        self.activeLeaseIDs = activeLeaseIDs
    }

    internal var isLeased: Bool {
        !activeLeaseIDs.isEmpty
    }
}

internal struct PromptCacheLease: Equatable, Sendable {
    let id: UUID
    let signature: PromptCacheSignature
    let tokens: [Int]
}

internal struct PromptCachePlan: @unchecked Sendable {
    let input: LMInput
    let cache: [KVCache]?
    let draftCache: [KVCache]?
    let reusedTokenCount: Int
    let lease: PromptCacheLease?

    internal init(
        input: LMInput,
        cache: [KVCache]?,
        draftCache: [KVCache]? = nil,
        reusedTokenCount: Int,
        lease: PromptCacheLease? = nil
    ) {
        self.input = input
        self.cache = cache
        self.draftCache = draftCache
        self.reusedTokenCount = reusedTokenCount
        self.lease = lease
    }
}

internal struct PromptCacheReusableState: @unchecked Sendable {
    let cache: [KVCache]
    let draftCache: [KVCache]?

    internal init(cache: [KVCache], draftCache: [KVCache]? = nil) {
        self.cache = cache
        self.draftCache = draftCache
    }
}

internal struct PromptCacheCandidate: Equatable, Sendable {
    let index: Int
    let reusableTokenCount: Int
}

internal struct PromptCacheReusePolicy: Sendable {
    let alignment: PromptCacheReuseAlignment
    let prefillStepSize: Int

    init(parameters: GenerateParameters) {
        self.init(
            alignment: parameters.promptCacheReuseAlignment,
            prefillStepSize: parameters.prefillStepSize
        )
    }

    init(alignment: PromptCacheReuseAlignment, prefillStepSize: Int) {
        self.alignment = alignment
        self.prefillStepSize = max(1, prefillStepSize)
    }

    func alignedReusableTokenCount(_ tokenCount: Int) -> Int {
        switch alignment {
        case .exact:
            return tokenCount

        case .prefillStep:
            return (tokenCount / prefillStepSize) * prefillStepSize
        }
    }

    func reusableTokenCount(
        commonPrefixCount: Int,
        requestTokenCount: Int,
        entryTokenCount: Int
    ) -> Int {
        let safeReusableTokenCount = min(
            commonPrefixCount,
            max(0, requestTokenCount - 1),
            entryTokenCount
        )
        guard safeReusableTokenCount > 0 else {
            return 0
        }

        if commonPrefixCount >= requestTokenCount {
            return safeReusableTokenCount
        }

        return alignedReusableTokenCount(safeReusableTokenCount)
    }

    func persistentPrefixTokenCount(
        cachedTokenCount: Int,
        requestTokenCount: Int,
        blockSize: Int
    ) -> Int {
        guard cachedTokenCount > 0 else {
            return 0
        }
        guard cachedTokenCount < requestTokenCount else {
            return cachedTokenCount
        }

        let alignedTokenCount = alignedReusableTokenCount(cachedTokenCount)
        switch alignment {
        case .exact:
            return alignedTokenCount

        case .prefillStep:
            let blockSize = max(1, blockSize)
            return (alignedTokenCount / blockSize) * blockSize
        }
    }
}

internal struct PromptCacheBlockEntry: @unchecked Sendable {
    let blockHash: String
    let blockIndex: Int
    let tokenRange: Range<Int>
    let tokens: [Int]
    let cache: [KVCache]
    let signature: PromptCacheSignature
    let byteCount: Int
}

private protocol PromptCacheLookupStrategy: Sendable {
    func bestCandidate(
        tokenIds: [Int],
        entries: [PromptCacheEntry],
        signature: PromptCacheSignature,
        reusePolicy: PromptCacheReusePolicy,
        requiresDraftCache: Bool
    ) -> PromptCacheCandidate?
}

private struct LinearPromptCacheLookupStrategy: PromptCacheLookupStrategy {
    func bestCandidate(
        tokenIds: [Int],
        entries: [PromptCacheEntry],
        signature: PromptCacheSignature,
        reusePolicy: PromptCacheReusePolicy,
        requiresDraftCache: Bool
    ) -> PromptCacheCandidate? {
        let candidate = entries
            .enumerated()
            .lazy
            .filter { $0.element.signature == signature }
            .filter { !requiresDraftCache || $0.element.draftCache != nil }
            .compactMap { index, entry -> PromptCacheCandidate? in
                PromptCachePlanner.candidate(
                    index: index,
                    entry: entry,
                    tokenIds: tokenIds,
                    reusePolicy: reusePolicy
                )
            }
            .max { lhs, rhs in
                lhs.reusableTokenCount < rhs.reusableTokenCount
            }
        recordLookup(candidate: candidate, candidateCount: entries.count)
        return candidate
    }

    private func recordLookup(candidate: PromptCacheCandidate?, candidateCount: Int) {
        MLXGenerationDiagnostics.recordPromptCacheLookup(MLXPromptCacheLookupSnapshot(
            strategy: .linear,
            blockSize: nil,
            matchedBlockCount: 0,
            candidateCount: candidateCount,
            selectedIndex: candidate?.index,
            reusedTokenCount: candidate?.reusableTokenCount ?? 0
        ))
    }
}

private struct BlockAwarePromptCacheLookupStrategy: PromptCacheLookupStrategy {
    private let blockSize: Int
    private let fallback: LinearPromptCacheLookupStrategy

    init(blockSize: Int = 256, fallback: LinearPromptCacheLookupStrategy = .init()) {
        self.blockSize = max(1, blockSize)
        self.fallback = fallback
    }

    func bestCandidate(
        tokenIds: [Int],
        entries: [PromptCacheEntry],
        signature: PromptCacheSignature,
        reusePolicy: PromptCacheReusePolicy,
        requiresDraftCache: Bool
    ) -> PromptCacheCandidate? {
        guard tokenIds.count >= blockSize else {
            return fallback.bestCandidate(
                tokenIds: tokenIds,
                entries: entries,
                signature: signature,
                reusePolicy: reusePolicy,
                requiresDraftCache: requiresDraftCache
            )
        }

        let index = PromptCacheBlockIndex(
            entries: entries,
            signature: signature,
            requiresDraftCache: requiresDraftCache,
            blockSize: blockSize
        )
        guard let lookup = index.lookup(tokenIds: tokenIds) else {
            return fallback.bestCandidate(
                tokenIds: tokenIds,
                entries: entries,
                signature: signature,
                reusePolicy: reusePolicy,
                requiresDraftCache: requiresDraftCache
            )
        }

        let candidate = lookup.candidateIndexes
            .compactMap { index -> PromptCacheCandidate? in
                guard entries.indices.contains(index) else {
                    return nil
                }
                return PromptCachePlanner.candidate(
                    index: index,
                    entry: entries[index],
                    tokenIds: tokenIds,
                    reusePolicy: reusePolicy
                )
            }
            .max { lhs, rhs in
                lhs.reusableTokenCount < rhs.reusableTokenCount
            }
        recordLookup(candidate: candidate, lookup: lookup)
        return candidate
    }

    private func recordLookup(
        candidate: PromptCacheCandidate?,
        lookup: PromptCacheBlockIndex.Lookup
    ) {
        MLXGenerationDiagnostics.recordPromptCacheLookup(MLXPromptCacheLookupSnapshot(
            strategy: .blockIndex,
            blockSize: blockSize,
            matchedBlockCount: lookup.matchedBlockCount,
            candidateCount: lookup.candidateIndexes.count,
            selectedIndex: candidate?.index,
            reusedTokenCount: candidate?.reusableTokenCount ?? 0
        ))
    }
}

internal struct PromptCacheBlockIndex: Sendable {
    internal struct Lookup: Equatable, Sendable {
        let matchedBlockCount: Int
        let candidateIndexes: [Int]
    }

    private let blockSize: Int
    private let entryIndexesByHash: [String: Set<Int>]

    internal init(
        entries: [PromptCacheEntry],
        signature: PromptCacheSignature,
        requiresDraftCache: Bool,
        blockSize: Int = 256
    ) {
        self.blockSize = max(1, blockSize)
        var entryIndexesByHash: [String: Set<Int>] = [:]
        for (index, entry) in entries.enumerated()
            where entry.signature == signature && (!requiresDraftCache || entry.draftCache != nil) {
            for hash in Self.prefixBlockHashes(for: entry.tokens, blockSize: self.blockSize) {
                entryIndexesByHash[hash, default: []].insert(index)
            }
        }
        self.entryIndexesByHash = entryIndexesByHash
    }

    internal func lookup(tokenIds: [Int]) -> Lookup? {
        let hashes = Self.prefixBlockHashes(for: tokenIds, blockSize: blockSize)
        guard !hashes.isEmpty else {
            return nil
        }
        for (offset, hash) in hashes.enumerated().reversed() {
            guard let indexes = entryIndexesByHash[hash], !indexes.isEmpty else {
                continue
            }
            return Lookup(
                matchedBlockCount: offset + 1,
                candidateIndexes: indexes.sorted()
            )
        }
        return nil
    }

    internal static func prefixBlockHashes(
        for tokenIds: [Int],
        blockSize: Int
    ) -> [String] {
        let blockSize = max(1, blockSize)
        let fullBlockCount = tokenIds.count / blockSize
        guard fullBlockCount > 0 else {
            return []
        }

        var priorHash = Data()
        var hashes: [String] = []
        hashes.reserveCapacity(fullBlockCount)

        for blockIndex in 0 ..< fullBlockCount {
            let startIndex = blockIndex * blockSize
            let endIndex = startIndex + blockSize
            let blockTokens = tokenIds[startIndex ..< endIndex]
            let hash = hashBlock(blockTokens, previousHash: priorHash)
            hashes.append(hash.hexString)
            priorHash = Data(hash)
        }

        return hashes
    }

    private static func hashBlock(
        _ tokens: ArraySlice<Int>,
        previousHash: Data
    ) -> SHA256.Digest {
        var hasher = SHA256()
        hasher.update(data: previousHash)
        var count = UInt64(tokens.count).littleEndian
        withUnsafeBytes(of: &count) { hasher.update(bufferPointer: $0) }
        for token in tokens {
            var value = Int64(token).littleEndian
            withUnsafeBytes(of: &value) { hasher.update(bufferPointer: $0) }
        }
        return hasher.finalize()
    }
}

internal enum PromptCachePlanner {
    private static let lookupStrategy: PromptCacheLookupStrategy =
        BlockAwarePromptCacheLookupStrategy()

    internal static func plan(
        fullInput: LMInput,
        tokenIds: [Int],
        parameters: GenerateParameters,
        cacheVariant: String? = nil,
        cacheLayout: [String]? = nil,
        promptCacheIdentity: PromptCacheIdentity? = nil,
        existingEntries: inout [PromptCacheEntry],
        reuseEnabled: Bool,
        requiresDraftCache: Bool = false
    ) -> PromptCachePlan {
        let noReuse = PromptCachePlan(input: fullInput, cache: nil, reusedTokenCount: 0)
        guard reuseEnabled, tokenIds.count > 1 else { return noReuse }
        guard let candidate = bestEntry(
            tokenIds: tokenIds,
            parameters: parameters,
            cacheVariant: cacheVariant,
            cacheLayout: cacheLayout,
            promptCacheIdentity: promptCacheIdentity,
            existingEntries: existingEntries,
            reusePolicy: PromptCacheReusePolicy(parameters: parameters),
            requiresDraftCache: requiresDraftCache
        ) else { return noReuse }

        let existingEntryIndex = candidate.index
        let existingEntry = existingEntries[existingEntryIndex]
        let reusableTokenCount = candidate.reusableTokenCount
        guard reusableTokenCount > 0 else { return noReuse }

        let cache = existingEntry.cache.map { $0.copy() }
        let draftCache = existingEntry.draftCache?.map { $0.copy() }
        if requiresDraftCache, draftCache == nil {
            return noReuse
        }
        let tokensToTrim = existingEntry.tokens.count - reusableTokenCount
        if tokensToTrim > 0 {
            guard canTrimPromptCache(cache) else { return noReuse }
            let trimmed = trimPromptCache(cache, numTokens: tokensToTrim)
            guard trimmed == tokensToTrim else { return noReuse }
            if let draftCache {
                guard canTrimPromptCache(draftCache) else { return noReuse }
                let draftTrimmed = trimPromptCache(draftCache, numTokens: tokensToTrim)
                guard draftTrimmed == tokensToTrim else { return noReuse }
            }
        }

        let suffixTokens = Array(tokenIds[reusableTokenCount...])
        guard let lease = acquireLease(at: existingEntryIndex, in: &existingEntries) else {
            return noReuse
        }
        return PromptCachePlan(
            input: LMInput(tokens: MLXArray(suffixTokens)),
            cache: cache,
            draftCache: draftCache,
            reusedTokenCount: reusableTokenCount,
            lease: lease
        )
    }

    private static func bestEntry(
        tokenIds: [Int],
        parameters: GenerateParameters,
        cacheVariant: String?,
        cacheLayout: [String]?,
        promptCacheIdentity: PromptCacheIdentity?,
        existingEntries: [PromptCacheEntry],
        reusePolicy: PromptCacheReusePolicy,
        requiresDraftCache: Bool
    ) -> PromptCacheCandidate? {
        let signature = PromptCacheSignature(
            parameters: parameters,
            cacheVariant: cacheVariant,
            cacheLayout: cacheLayout,
            promptCacheIdentity: promptCacheIdentity
        )
        return lookupStrategy.bestCandidate(
            tokenIds: tokenIds,
            entries: existingEntries,
            signature: signature,
            reusePolicy: reusePolicy,
            requiresDraftCache: requiresDraftCache
        )
    }

    internal static func bestCandidate(
        tokenIds: [Int],
        parameters: GenerateParameters,
        cacheVariant: String? = nil,
        cacheLayout: [String]? = nil,
        promptCacheIdentity: PromptCacheIdentity? = nil,
        existingEntries: [PromptCacheEntry],
        requiresDraftCache: Bool = false
    ) -> PromptCacheCandidate? {
        bestEntry(
            tokenIds: tokenIds,
            parameters: parameters,
            cacheVariant: cacheVariant,
            cacheLayout: cacheLayout,
            promptCacheIdentity: promptCacheIdentity,
            existingEntries: existingEntries,
            reusePolicy: PromptCacheReusePolicy(parameters: parameters),
            requiresDraftCache: requiresDraftCache
        )
    }

    internal static func candidate(
        index: Int,
        entry: PromptCacheEntry,
        tokenIds: [Int]
    ) -> PromptCacheCandidate? {
        candidate(
            index: index,
            entry: entry,
            tokenIds: tokenIds,
            reusePolicy: PromptCacheReusePolicy(alignment: .exact, prefillStepSize: 1)
        )
    }

    fileprivate static func candidate(
        index: Int,
        entry: PromptCacheEntry,
        tokenIds: [Int],
        reusePolicy: PromptCacheReusePolicy
    ) -> PromptCacheCandidate? {
        let commonPrefix = commonPrefixCount(entry.tokens, tokenIds)
        let reusableTokenCount = reusePolicy.reusableTokenCount(
            commonPrefixCount: commonPrefix,
            requestTokenCount: tokenIds.count,
            entryTokenCount: entry.tokens.count
        )
        guard reusableTokenCount > 0 else { return nil }
        return PromptCacheCandidate(
            index: index,
            reusableTokenCount: reusableTokenCount
        )
    }

    internal static func makeEntry(
        tokenIds: [Int],
        cache: [KVCache],
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        cacheVariant: String? = nil,
        promptCacheIdentity: PromptCacheIdentity? = nil,
        maxBytes: Int?
    ) -> PromptCacheEntry? {
        guard !tokenIds.isEmpty else { return nil }

        guard let snapshot = copyCache(cache, trimmingToTokenCount: tokenIds.count) else {
            return nil
        }
        eval(snapshot)
        let draftSnapshot: [KVCache]?
        if let draftCache {
            guard let copiedDraftCache = copyCache(
                draftCache,
                trimmingToTokenCount: tokenIds.count
            ) else {
                return nil
            }
            draftSnapshot = copiedDraftCache
        } else {
            draftSnapshot = nil
        }
        if let draftSnapshot {
            eval(draftSnapshot)
        }

        let byteCount = cacheByteCount(snapshot) + cacheByteCount(draftSnapshot ?? [])
        if let maxBytes, byteCount > maxBytes {
            return nil
        }

        return PromptCacheEntry(
            tokens: tokenIds,
            cache: snapshot,
            draftCache: draftSnapshot,
            signature: PromptCacheSignature(
                parameters: parameters,
                cacheVariant: cacheVariant,
                cacheLayout: cacheLayoutFingerprint(for: snapshot, draftCache: draftSnapshot),
                promptCacheIdentity: promptCacheIdentity
            ),
            byteCount: byteCount
        )
    }

    internal static func store(
        _ entry: PromptCacheEntry,
        in entries: inout [PromptCacheEntry],
        maxBytes: Int?,
        maxEntries: Int = 4
    ) {
        entries.removeAll {
            !$0.isLeased && $0.signature == entry.signature && $0.tokens == entry.tokens
        }
        entries.insert(entry, at: 0)

        var totalBytes = entries.reduce(0) { $0 + $1.byteCount }
        while entries.count > maxEntries || maxBytes.map({ totalBytes > $0 }) == true {
            guard let removableIndex = entries.lastIndex(where: { !$0.isLeased }) else {
                break
            }
            let removed = entries.remove(at: removableIndex)
            totalBytes -= removed.byteCount
        }
    }

    internal static func acquireLease(
        at index: Int,
        in entries: inout [PromptCacheEntry]
    ) -> PromptCacheLease? {
        guard entries.indices.contains(index) else {
            return nil
        }
        let entry = entries[index]
        let lease = PromptCacheLease(
            id: UUID(),
            signature: entry.signature,
            tokens: entry.tokens
        )
        entries[index].activeLeaseIDs.insert(lease.id)
        return lease
    }

    internal static func release(
        _ lease: PromptCacheLease,
        in entries: inout [PromptCacheEntry]
    ) {
        guard let index = entries.firstIndex(where: { entry in
            entry.signature == lease.signature && entry.tokens == lease.tokens
        }) else {
            return
        }
        entries[index].activeLeaseIDs.remove(lease.id)
    }

    private static func commonPrefixCount(_ lhs: [Int], _ rhs: [Int]) -> Int {
        var count = 0
        for (left, right) in zip(lhs, rhs) {
            guard left == right else { break }
            count += 1
        }
        return count
    }

    internal static func cacheByteCount(_ cache: [KVCache]) -> Int {
        cache.reduce(into: 0) { total, cache in
            total += cache.innerState().reduce(0) { $0 + $1.nbytes }
        }
    }

    internal static func copyCache(
        _ cache: [KVCache],
        trimmingToTokenCount tokenCount: Int
    ) -> [KVCache]? {
        let snapshot = cache.map { $0.copy() }
        guard trimCache(snapshot, toTokenCount: tokenCount) else { return nil }
        return snapshot
    }

    internal static func copyCache(
        _ cache: [KVCache],
        trimmingToTokenRange tokenRange: Range<Int>
    ) -> [KVCache]? {
        guard tokenRange.lowerBound >= 0,
            tokenRange.lowerBound <= tokenRange.upperBound,
            let snapshot = copyCache(cache, trimmingToTokenCount: tokenRange.upperBound)
        else {
            return nil
        }
        guard trimCacheFront(snapshot, droppingTokenCount: tokenRange.lowerBound) else {
            return nil
        }
        return snapshot
    }

    internal static func makeBlockEntries(
        from entry: PromptCacheEntry,
        blockSize: Int = 256,
        maxBlockBytes: Int? = nil
    ) -> [PromptCacheBlockEntry] {
        guard entry.draftCache == nil else {
            return []
        }

        let blockSize = max(1, blockSize)
        let hashes = PromptCacheBlockIndex.prefixBlockHashes(
            for: entry.tokens,
            blockSize: blockSize
        )
        guard !hashes.isEmpty else {
            return []
        }

        return hashes.enumerated().compactMap { blockIndex, blockHash in
            let startIndex = blockIndex * blockSize
            let tokenRange = startIndex ..< startIndex + blockSize
            guard let cache = copyCache(entry.cache, trimmingToTokenRange: tokenRange) else {
                return nil
            }

            let byteCount = cacheByteCount(cache)
            guard maxBlockBytes.map({ byteCount <= $0 }) ?? true else {
                return nil
            }

            return PromptCacheBlockEntry(
                blockHash: blockHash,
                blockIndex: blockIndex,
                tokenRange: tokenRange,
                tokens: Array(entry.tokens[tokenRange]),
                cache: cache,
                signature: entry.signature,
                byteCount: byteCount
            )
        }
    }

    private static func trimCache(_ cache: [KVCache], toTokenCount tokenCount: Int) -> Bool {
        guard !cache.isEmpty else { return true }
        let reusableTokenCount = cache.map(effectiveOffset).max() ?? 0
        guard reusableTokenCount >= tokenCount else { return false }

        let tokensToTrim = reusableTokenCount - tokenCount
        guard tokensToTrim > 0 else { return true }
        guard canTrimPromptCache(cache) else { return false }

        let trimmed = trimPromptCache(cache, numTokens: tokensToTrim)
        guard trimmed == tokensToTrim else { return false }
        return cache.map(effectiveOffset).allSatisfy { $0 <= tokenCount }
    }

    private static func trimCacheFront(
        _ cache: [KVCache],
        droppingTokenCount tokenCount: Int
    ) -> Bool {
        guard tokenCount > 0 else {
            return true
        }
        return cache.allSatisfy { cache in
            trimCacheFront(cache, droppingTokenCount: tokenCount)
        }
    }

    private static func trimCacheFront(
        _ cache: KVCache,
        droppingTokenCount tokenCount: Int
    ) -> Bool {
        if let cacheList = cache as? CacheList {
            return cacheList.layoutCaches.allSatisfy { childCache in
                trimCacheFront(childCache, droppingTokenCount: tokenCount)
            }
        }

        var cache = cache
        let targetOffset = cache.offset - tokenCount
        guard targetOffset >= 0 else {
            return false
        }

        let state = cache.state
        guard !state.isEmpty else {
            return cache.offset == targetOffset
        }

        var slicedState: [MLXArray] = []
        slicedState.reserveCapacity(state.count)
        for array in state {
            guard array.ndim > 2, array.dim(2) >= tokenCount else {
                return false
            }
            slicedState.append(array[.ellipsis, tokenCount..., 0...])
        }
        cache.state = slicedState
        return setFrontTrimmedMetadata(cache, targetOffset: targetOffset)
    }

    private static func setFrontTrimmedMetadata(
        _ cache: KVCache,
        targetOffset: Int
    ) -> Bool {
        switch cache {
        case let quantizedRotatingCache as QuantizedRotatingKVCache:
            var metadata = quantizedRotatingCache.metaState
            guard metadata.count == 8 else {
                return false
            }
            metadata[3] = String(targetOffset)
            metadata[4] = String(min(targetOffset, Int(metadata[1]) ?? targetOffset))
            quantizedRotatingCache.metaState = metadata
            return quantizedRotatingCache.offset == targetOffset

        case let rotatingCache as RotatingKVCache:
            var metadata = rotatingCache.metaState
            guard metadata.count == 5 else {
                return false
            }
            metadata[3] = String(targetOffset)
            metadata[4] = String(min(targetOffset, Int(metadata[1]) ?? targetOffset))
            rotatingCache.metaState = metadata
            return rotatingCache.offset == targetOffset

        case let quantizedCache as QuantizedKVCache:
            var metadata = quantizedCache.metaState
            guard metadata.count == 4 else {
                return false
            }
            metadata[1] = String(targetOffset)
            quantizedCache.metaState = metadata
            return quantizedCache.offset == targetOffset

        case let chunkedCache as ChunkedKVCache:
            var metadata = chunkedCache.metaState
            guard metadata.count == 2 else {
                return false
            }
            metadata[1] = "0"
            chunkedCache.metaState = metadata
            return chunkedCache.offset == targetOffset

        case is MambaCache:
            return false

        default:
            return cache.offset == targetOffset
        }
    }

    private static func effectiveOffset(for cache: KVCache) -> Int {
        guard let cacheList = cache as? CacheList else {
            return cache.offset
        }
        return cacheList.layoutCaches.map(effectiveOffset).max() ?? cache.offset
    }

    internal static func cacheLayoutFingerprint(
        for cache: [KVCache],
        draftCache: [KVCache]? = nil
    ) -> [String] {
        var components = cache.enumerated().map { index, cache in
            "main[\(index)]:\(cacheLayoutSignature(for: cache))"
        }
        if let draftCache {
            components.append(contentsOf: draftCache.enumerated().map { index, cache in
                "draft[\(index)]:\(cacheLayoutSignature(for: cache))"
            })
        }
        return components
    }

    private static func cacheLayoutSignature(for cache: KVCache) -> String {
        switch cache {
        case let cacheList as CacheList:
            let children = cacheList.layoutCaches
                .map { cacheLayoutSignature(for: $0) }
                .joined(separator: ",")
            return "CacheList[\(children)]"
        case let quantizedRotatingCache as QuantizedRotatingKVCache:
            let metadata = quantizedRotatingCache.metaState
            let keep = metadata[safe: 0] ?? "0"
            let maxSize = metadata[safe: 1] ?? quantizedRotatingCache.maxSize.map(String.init) ?? "nil"
            let step = metadata[safe: 2] ?? "unknown"
            return "QuantizedRotatingKVCache(keep:\(keep),maxSize:\(maxSize),step:\(step)," +
                "groupSize:\(quantizedRotatingCache.groupSize)," +
                "bits:\(quantizedRotatingCache.bits),mode:\(quantizedRotatingCache.mode))"
        case let rotatingCache as RotatingKVCache:
            let metadata = rotatingCache.metaState
            let keep = metadata[safe: 0] ?? "0"
            let maxSize = metadata[safe: 1] ?? rotatingCache.maxSize.map(String.init) ?? "nil"
            let step = metadata[safe: 2] ?? "unknown"
            return "RotatingKVCache(keep:\(keep),maxSize:\(maxSize),step:\(step))"
        case let quantizedCache as QuantizedKVCacheProtocol:
            return "QuantizedKVCache(groupSize:\(quantizedCache.groupSize)," +
                "bits:\(quantizedCache.bits),mode:\(quantizedCache.mode))"
        case is MiniMaxM3BatchKVCache:
            return "MiniMaxM3BatchKVCache(fullState:true)"
        case is MiniMaxM3KVCache:
            return "MiniMaxM3KVCache(indexKeys:true)"
        case is ChunkedKVCache:
            return "ChunkedKVCache(maxSize:\(cache.maxSize.map(String.init) ?? "nil"))"
        case is MambaCache:
            return "MambaCache"
        case is KVCacheSimple:
            return "KVCache"
        default:
            return String(describing: type(of: cache))
        }
    }
}

private extension SHA256.Digest {
    var hexString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}

private extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
