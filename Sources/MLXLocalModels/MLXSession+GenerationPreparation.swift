import Foundation
import MLX

extension MLXSession {
    private struct PreparedPrompt {
        let fullInput: LMInput
        let tokenIDs: [Int]
        let startTime: ContinuousClock.Instant
    }

    private struct CacheOptions {
        let speculativeDecoding: MLXSpeculativeDecodingConfiguration?
        let promptCacheVariant: String?

        var requiresDraftCache: Bool {
            speculativeDecoding != nil
        }
    }

    private struct CachePreparation {
        let options: CacheOptions
        let plan: PromptCachePlan
    }

    nonisolated func prepareGeneration(
        genContext: GenerationContext,
        promptCacheEntries: inout [PromptCacheEntry],
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?,
        promptCacheVariant: String?
    ) throws -> MLXPreparedGeneration {
        let prompt = try tokenizedPrompt(genContext: genContext)
        let memorySnapshot = MLXRuntimeMemorySnapshot.live()
        let adaptedContext = adaptedGenerationContext(
            genContext: genContext,
            prompt: prompt,
            memorySnapshot: memorySnapshot
        )
        let cache = cachePreparation(
            genContext: adaptedContext,
            promptCacheEntries: &promptCacheEntries,
            prompt: prompt,
            speculativeDecoding: speculativeDecoding,
            promptCacheVariant: promptCacheVariant
        )
        try preflightGeneration(
            genContext: adaptedContext,
            promptTokenCount: prompt.tokenIDs.count,
            cachePlan: cache.plan,
            memorySnapshot: memorySnapshot
        )
        return preparedGeneration(
            genContext: adaptedContext,
            prompt: prompt,
            cachePlan: cache.plan,
            cacheOptions: cache.options,
            memorySnapshot: memorySnapshot
        )
    }

    nonisolated private func adaptedGenerationContext(
        genContext: GenerationContext,
        prompt: PreparedPrompt,
        memorySnapshot: MLXRuntimeMemorySnapshot
    ) -> GenerationContext {
        adaptivePrefillContext(
            genContext: genContext,
            promptTokenCount: prompt.tokenIDs.count,
            memorySnapshot: memorySnapshot
        )
    }

    nonisolated private func adaptivePrefillContext(
        genContext: GenerationContext,
        promptTokenCount: Int,
        memorySnapshot: MLXRuntimeMemorySnapshot
    ) -> GenerationContext {
        let kvBits = genContext.runtimePreferences.optimization.kvCacheBitsForMemoryGuard
        let memoryProfile = genContext.memoryProfile?.applyingKVCacheBits(kvBits)
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: genContext.runtimePreferences.memoryGuard,
            profile: memoryProfile,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: 0,
            requestedChunkSize: genContext.parameters.prefillStepSize,
            memorySnapshot: memorySnapshot
        )
        MLXGenerationDiagnostics.recordAdaptivePrefillChunk(decision.snapshot)
        guard decision.selectedChunkSize != genContext.parameters.prefillStepSize else {
            return genContext
        }

        var parameters = genContext.parameters
        parameters.prefillStepSize = decision.selectedChunkSize
        return GenerationContext(
            modelContext: genContext.modelContext,
            input: genContext.input,
            parameters: parameters,
            generationStartTime: genContext.generationStartTime,
            continuation: genContext.continuation,
            clock: genContext.clock,
            runtimePreferences: genContext.runtimePreferences,
            memoryProfile: genContext.memoryProfile
        )
    }

    nonisolated private func tokenizedPrompt(
        genContext: GenerationContext
    ) throws -> PreparedPrompt {
        let promptStartTime = genContext.clock.now
        let fullInput = try genContext.modelContext.tokenize(input: genContext.input)
        eval(fullInput.text.tokens)
        let tokenizationEndTime = genContext.clock.now
        let tokenIDs = fullInput.text.tokens.asArray(Int.self)
        let duration = promptStartTime.duration(to: tokenizationEndTime)
        logTokenizationDuration(duration, tokenCount: tokenIDs.count)
        return PreparedPrompt(
            fullInput: fullInput,
            tokenIDs: tokenIDs,
            startTime: promptStartTime
        )
    }

    nonisolated private func cachePreparation(
        genContext: GenerationContext,
        promptCacheEntries: inout [PromptCacheEntry],
        prompt: PreparedPrompt,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?,
        promptCacheVariant: String?
    ) -> CachePreparation {
        let options = CacheOptions(
            speculativeDecoding: speculativeDecoding,
            promptCacheVariant: promptCacheVariant
        )
        let plan = promptCachePlan(
            genContext: genContext,
            promptCacheEntries: &promptCacheEntries,
            prompt: prompt,
            cacheOptions: options
        )
        MLXGenerationDiagnostics.recordPromptCachePlan(
            promptTokenCount: prompt.tokenIDs.count,
            reusedTokenCount: plan.reusedTokenCount
        )
        recordSpecPrefillFallbackIfNeeded(
            genContext: genContext,
            promptTokenCount: prompt.tokenIDs.count,
            cachePlan: plan
        )
        return CachePreparation(options: options, plan: plan)
    }

    nonisolated private func recordSpecPrefillFallbackIfNeeded(
        genContext: GenerationContext,
        promptTokenCount: Int,
        cachePlan: PromptCachePlan
    ) {
        guard genContext.runtimePreferences.optimization.mode == .specPrefill else {
            return
        }

        MLXSpecPrefillPlanner.recordRuntimeUnavailable(
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachePlan.reusedTokenCount,
            protectedPrefixTokenCount: cachePlan.reusedTokenCount,
            configuration: genContext.runtimePreferences.optimization.specPrefill
        )
    }

    nonisolated private func promptCachePlan(
        genContext: GenerationContext,
        promptCacheEntries: inout [PromptCacheEntry],
        prompt: PreparedPrompt,
        cacheOptions: CacheOptions
    ) -> PromptCachePlan {
        let draftCache = cacheOptions.speculativeDecoding?.draftContext.model.newCache(
            parameters: genContext.parameters
        )
        let cacheLayout = PromptCachePlanner.cacheLayoutFingerprint(
            for: genContext.modelContext.model.newCache(parameters: genContext.parameters),
            draftCache: draftCache
        )
        restorePersistentPromptCache(
            genContext: genContext,
            promptCacheEntries: &promptCacheEntries,
            prompt: prompt,
            cacheOptions: cacheOptions,
            cacheLayout: cacheLayout
        )
        return PromptCachePlanner.plan(
            fullInput: prompt.fullInput,
            tokenIds: prompt.tokenIDs,
            parameters: genContext.parameters,
            cacheVariant: cacheOptions.promptCacheVariant,
            cacheLayout: cacheLayout,
            promptCacheIdentity: genContext.input.promptCacheIdentity,
            existingEntries: &promptCacheEntries,
            reuseEnabled: genContext.input.limits.reusePromptCache,
            requiresDraftCache: cacheOptions.requiresDraftCache
        )
    }

    nonisolated private func restorePersistentPromptCache(
        genContext: GenerationContext,
        promptCacheEntries: inout [PromptCacheEntry],
        prompt: PreparedPrompt,
        cacheOptions: CacheOptions,
        cacheLayout: [String]
    ) {
        let signature = PromptCacheSignature(
            parameters: genContext.parameters,
            cacheVariant: cacheOptions.promptCacheVariant,
            cacheLayout: cacheLayout,
            promptCacheIdentity: genContext.input.promptCacheIdentity
        )
        restorePersistentPromptCacheSnapshotIfNeeded(
            tokenIds: prompt.tokenIDs,
            signature: signature,
            promptCacheEntries: &promptCacheEntries,
            genContext: genContext,
            speculativeDecoding: cacheOptions.speculativeDecoding
        )
    }

    nonisolated private func preflightGeneration(
        genContext: GenerationContext,
        promptTokenCount: Int,
        cachePlan: PromptCachePlan,
        memorySnapshot: MLXRuntimeMemorySnapshot
    ) throws {
        let kvBits = genContext.runtimePreferences.optimization.kvCacheBitsForMemoryGuard
        let memoryProfile = genContext.memoryProfile?.applyingKVCacheBits(kvBits)
        try MLXRuntimeMemoryGuard.preflight(
            configuration: genContext.runtimePreferences.memoryGuard,
            profile: memoryProfile,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachePlan.reusedTokenCount,
            maximumGeneratedTokenCount: genContext.input.limits.maxTokens,
            prefillStepSize: genContext.parameters.prefillStepSize,
            memorySnapshot: memorySnapshot
        )
    }

    nonisolated private func preparedGeneration(
        genContext: GenerationContext,
        prompt: PreparedPrompt,
        cachePlan: PromptCachePlan,
        cacheOptions: CacheOptions,
        memorySnapshot: MLXRuntimeMemorySnapshot
    ) -> MLXPreparedGeneration {
        let state = GenerationState()
        let tokenContext = TokenContext(
            state: state,
            context: genContext.modelContext,
            input: genContext.input,
            continuation: genContext.continuation,
            clock: genContext.clock,
            stopTokenIDs: stopTokenIDs(for: genContext.modelContext)
        )
        state.stopDetector = StopSequenceDetector(sequences: genContext.input.sampling.stopSequences)
        state.detokenizer = NaiveStreamingDetokenizer(tokenizer: genContext.modelContext.tokenizer)
        return MLXPreparedGeneration(
            genContext: genContext,
            fullInput: prompt.fullInput,
            promptTokenIDs: prompt.tokenIDs,
            cachePlan: cachePlan,
            state: state,
            tokenContext: tokenContext,
            promptStartTime: prompt.startTime,
            usesSpeculativeDecoding: cacheOptions.speculativeDecoding != nil,
            adaptivePrefillController: adaptivePrefillController(
                genContext: genContext,
                promptTokenCount: prompt.tokenIDs.count,
                cachedTokenCount: cachePlan.reusedTokenCount,
                memorySnapshot: memorySnapshot
            )
        )
    }

    nonisolated private func adaptivePrefillController(
        genContext: GenerationContext,
        promptTokenCount: Int,
        cachedTokenCount: Int,
        memorySnapshot: MLXRuntimeMemorySnapshot
    ) -> MLXAdaptivePrefillChunkController? {
        guard genContext.runtimePreferences.memoryGuard.tier != .off else {
            return nil
        }
        let kvBits = genContext.runtimePreferences.optimization.kvCacheBitsForMemoryGuard
        guard let memoryProfile = genContext.memoryProfile?.applyingKVCacheBits(kvBits) else {
            return nil
        }
        return MLXAdaptivePrefillChunkController(
            configuration: genContext.runtimePreferences.memoryGuard,
            profile: memoryProfile,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount,
            requestedChunkSize: genContext.parameters.prefillStepSize,
            memorySnapshot: memorySnapshot
        )
    }

    nonisolated private func logTokenizationDuration(
        _ duration: Duration,
        tokenCount: Int
    ) {
        guard duration > .seconds(10) else {
            return
        }
        logger.warning("Slow tokenization: \(duration) for \(tokenCount) tokens")
    }

    nonisolated private func stopTokenIDs(for context: ModelContext) -> Set<Int> {
        var tokenIDs = context.configuration.eosTokenIds
        if let unknownTokenID = context.tokenizer.unknownTokenId {
            tokenIDs.insert(unknownTokenID)
        }
        if let eosTokenID = context.tokenizer.eosTokenId {
            tokenIDs.insert(eosTokenID)
        }
        tokenIDs.formUnion(
            context.configuration.extraEOSTokens.compactMap { eosToken in
                context.tokenizer.convertTokenToId(eosToken)
            }
        )
        return tokenIDs
    }
}
