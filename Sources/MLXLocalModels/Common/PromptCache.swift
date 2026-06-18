import Foundation
import MLX

internal struct PromptCacheSignature: Codable, Equatable, Sendable {
    let cacheVariant: String?
    let maxKVSize: Int?
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let prefillStepSize: Int
    let promptCacheIdentity: PromptCacheIdentity?

    init(
        parameters: GenerateParameters,
        cacheVariant: String? = nil,
        promptCacheIdentity: PromptCacheIdentity? = nil
    ) {
        self.cacheVariant = cacheVariant
        self.maxKVSize = parameters.maxKVSize
        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.prefillStepSize = parameters.prefillStepSize
        self.promptCacheIdentity = promptCacheIdentity
    }
}

internal struct PromptCacheEntry: @unchecked Sendable {
    let tokens: [Int]
    let cache: [KVCache]
    let draftCache: [KVCache]?
    let signature: PromptCacheSignature
    let byteCount: Int

    internal init(
        tokens: [Int],
        cache: [KVCache],
        draftCache: [KVCache]? = nil,
        signature: PromptCacheSignature,
        byteCount: Int
    ) {
        self.tokens = tokens
        self.cache = cache
        self.draftCache = draftCache
        self.signature = signature
        self.byteCount = byteCount
    }
}

internal struct PromptCachePlan: @unchecked Sendable {
    let input: LMInput
    let cache: [KVCache]?
    let draftCache: [KVCache]?
    let reusedTokenCount: Int

    internal init(
        input: LMInput,
        cache: [KVCache]?,
        draftCache: [KVCache]? = nil,
        reusedTokenCount: Int
    ) {
        self.input = input
        self.cache = cache
        self.draftCache = draftCache
        self.reusedTokenCount = reusedTokenCount
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

internal enum PromptCachePlanner {
    internal static func plan(
        fullInput: LMInput,
        tokenIds: [Int],
        parameters: GenerateParameters,
        cacheVariant: String? = nil,
        promptCacheIdentity: PromptCacheIdentity? = nil,
        existingEntries: [PromptCacheEntry],
        reuseEnabled: Bool,
        requiresDraftCache: Bool = false
    ) -> PromptCachePlan {
        let noReuse = PromptCachePlan(input: fullInput, cache: nil, reusedTokenCount: 0)
        guard reuseEnabled, tokenIds.count > 1 else { return noReuse }
        guard let candidate = bestEntry(
            tokenIds: tokenIds,
            parameters: parameters,
            cacheVariant: cacheVariant,
            promptCacheIdentity: promptCacheIdentity,
            existingEntries: existingEntries,
            requiresDraftCache: requiresDraftCache
        ) else { return noReuse }

        let existingEntry = candidate.entry
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
        return PromptCachePlan(
            input: LMInput(tokens: MLXArray(suffixTokens)),
            cache: cache,
            draftCache: draftCache,
            reusedTokenCount: reusableTokenCount
        )
    }

    private static func bestEntry(
        tokenIds: [Int],
        parameters: GenerateParameters,
        cacheVariant: String?,
        promptCacheIdentity: PromptCacheIdentity?,
        existingEntries: [PromptCacheEntry],
        requiresDraftCache: Bool
    ) -> (entry: PromptCacheEntry, reusableTokenCount: Int)? {
        let signature = PromptCacheSignature(
            parameters: parameters,
            cacheVariant: cacheVariant,
            promptCacheIdentity: promptCacheIdentity
        )
        return existingEntries
            .lazy
            .filter { $0.signature == signature }
            .filter { !requiresDraftCache || $0.draftCache != nil }
            .compactMap { entry -> (entry: PromptCacheEntry, reusableTokenCount: Int)? in
                let commonPrefix = commonPrefixCount(entry.tokens, tokenIds)
                let reusableTokenCount = min(commonPrefix, tokenIds.count - 1, entry.tokens.count)
                guard reusableTokenCount > 0 else { return nil }
                return (entry, reusableTokenCount)
            }
            .max { lhs, rhs in
                lhs.reusableTokenCount < rhs.reusableTokenCount
            }
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

        let snapshot = cache.map { $0.copy() }
        eval(snapshot)
        let draftSnapshot = draftCache?.map { $0.copy() }
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
            $0.signature == entry.signature && $0.tokens == entry.tokens
        }
        entries.insert(entry, at: 0)

        var totalBytes = entries.reduce(0) { $0 + $1.byteCount }
        while entries.count > maxEntries || maxBytes.map({ totalBytes > $0 }) == true {
            let removed = entries.removeLast()
            totalBytes -= removed.byteCount
        }
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
}
