extension MLXSession {
    nonisolated func updatePromptCacheEntries(
        _ entries: inout [PromptCacheEntry],
        tokenIds: [Int],
        reusableState: PromptCacheReusableState,
        genContext: GenerationContext,
        promptCacheVariant: String?
    ) {
        guard genContext.input.limits.reusePromptCache else {
            entries.removeAll()
            return
        }

        guard let entry = PromptCachePlanner.makeEntry(
            tokenIds: tokenIds,
            cache: reusableState.cache,
            draftCache: reusableState.draftCache,
            parameters: genContext.parameters,
            cacheVariant: promptCacheVariant,
            promptCacheIdentity: genContext.input.promptCacheIdentity,
            maxBytes: genContext.input.limits.maxPromptCacheBytes
        ) else {
            return
        }

        PromptCachePlanner.store(
            entry,
            in: &entries,
            maxBytes: genContext.input.limits.maxPromptCacheBytes
        )
    }
}
