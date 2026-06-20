import Foundation

internal struct MLXPreparedGeneration {
    internal let genContext: GenerationContext
    internal let fullInput: LMInput
    internal let promptTokenIDs: [Int]
    internal let cachePlan: PromptCachePlan
    internal let state: GenerationState
    internal let tokenContext: TokenContext
    internal let promptStartTime: ContinuousClock.Instant
    internal let usesSpeculativeDecoding: Bool
    internal let adaptivePrefillController: MLXAdaptivePrefillChunkController?

    internal func makeContinuousBatchPrefillRequest(
        sink: MLXContinuousBatchStreamSink,
        promptCacheVariant: String? = nil
    ) throws -> MLXContinuousBatchPrefillRequest {
        try validateContinuousBatchingSupport()
        return MLXContinuousBatchPrefillRequest(
            promptTokenIDs: try continuousBatchPrefillTokenIDs(),
            parameters: genContext.parameters,
            state: state,
            tokenizer: genContext.modelContext.tokenizer,
            sink: sink,
            stopTokenIDs: continuousBatchStopTokenIDs(),
            processorPromptTokenIDs: promptTokenIDs,
            prefixCache: continuousBatchPrefixCache(),
            promptCacheStorage: continuousBatchPromptCacheStorage(
                promptCacheVariant: promptCacheVariant
            )
        )
    }

    private func validateContinuousBatchingSupport() throws {
        if usesSpeculativeDecoding {
            throw MLXPreparedGenerationBatchingError.speculativeDecodingUnsupported
        }
        if cachePlan.draftCache != nil || missingReusableCache {
            throw MLXPreparedGenerationBatchingError.promptCacheReuseUnsupported(
                reusedTokenCount: cachePlan.reusedTokenCount
            )
        }
        if promptTokenIDs.isEmpty {
            throw MLXPreparedGenerationBatchingError.emptyPrompt
        }
    }

    private var missingReusableCache: Bool {
        cachePlan.reusedTokenCount > 0 && cachePlan.cache == nil
    }

    private func continuousBatchPrefillTokenIDs() throws -> [Int] {
        if cachePlan.reusedTokenCount > 0 {
            let tokenIDs = cachePlan.input.text.tokens.asArray(Int.self)
            guard !tokenIDs.isEmpty else {
                throw MLXPreparedGenerationBatchingError.emptyPrompt
            }
            return tokenIDs
        }
        return promptTokenIDs
    }

    private func continuousBatchPrefixCache() -> MLXContinuousBatchPrefixCache? {
        guard let cache = cachePlan.cache, cachePlan.reusedTokenCount > 0 else {
            return nil
        }
        return MLXContinuousBatchPrefixCache(
            caches: cache,
            cachedTokenCount: cachePlan.reusedTokenCount
        )
    }

    private func continuousBatchPromptCacheStorage(
        promptCacheVariant: String?
    ) -> MLXContinuousBatchPromptCacheStorage {
        MLXContinuousBatchPromptCacheStorage(
            tokenIDs: promptTokenIDs,
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

    private func continuousBatchStopTokenIDs() -> Set<Int> {
        var stopTokenIDs = genContext.modelContext.configuration.eosTokenIds
        stopTokenIDs.formUnion(tokenizerStopTokenIDs())
        stopTokenIDs.formUnion(extraStopTokenIDs())
        return stopTokenIDs
    }

    private func tokenizerStopTokenIDs() -> Set<Int> {
        [
            genContext.modelContext.tokenizer.unknownTokenId,
            genContext.modelContext.tokenizer.eosTokenId
        ].compactMap(\.self).reduce(into: []) { tokenIDs, tokenID in
            tokenIDs.insert(tokenID)
        }
    }

    private func extraStopTokenIDs() -> Set<Int> {
        genContext.modelContext.configuration.extraEOSTokens.reduce(into: []) { tokenIDs, token in
            if let tokenID = genContext.modelContext.tokenizer.convertTokenToId(token) {
                tokenIDs.insert(tokenID)
            }
        }
    }
}
