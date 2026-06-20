extension MLXSession {
    nonisolated func updatePromptCacheEntries(
        _ entries: inout [PromptCacheEntry],
        tokenIds: [Int],
        reusableState: PromptCacheReusableState,
        genContext: GenerationContext,
        promptCacheVariant: String?
    ) {
        MLXPromptCacheEntryStore.update(
            &entries,
            tokenIDs: tokenIds,
            reusableState: reusableState,
            request: MLXPromptCacheEntryStore.Request(
                parameters: genContext.parameters,
                cacheVariant: promptCacheVariant,
                promptCacheIdentity: genContext.input.promptCacheIdentity,
                maxBytes: genContext.input.limits.maxPromptCacheBytes,
                reusePromptCache: genContext.input.limits.reusePromptCache,
                runtimePreferences: genContext.runtimePreferences
            )
        )
    }

    nonisolated func restorePersistentPromptCacheSnapshotIfNeeded(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        promptCacheEntries: inout [PromptCacheEntry],
        genContext: GenerationContext,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    ) {
        guard genContext.runtimePreferences.promptCachePolicy == .persistent,
            speculativeDecoding == nil,
            genContext.input.limits.reusePromptCache
        else {
            return
        }

        guard let entry = restorePersistentPromptCacheEntry(
            tokenIds: tokenIds,
            signature: signature,
            parameters: genContext.parameters,
            maxBytes: genContext.input.limits.maxPromptCacheBytes
        ) else {
            return
        }
        guard !promptCacheEntries.contains(where: { existing in
            existing.signature == entry.signature && existing.tokens == entry.tokens
        }) else {
            return
        }
        promptCacheEntries.insert(entry, at: 0)
    }

    nonisolated private func restorePersistentPromptCacheEntry(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        parameters: GenerateParameters,
        maxBytes: Int?
    ) -> PromptCacheEntry? {
        MLXPersistentPromptCacheRestorer.restoreBestEntry(
            tokenIds: tokenIds,
            signature: signature,
            reusePolicy: PromptCacheReusePolicy(parameters: parameters),
            maxBytes: maxBytes
        )
    }
}
