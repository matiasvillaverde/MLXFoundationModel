import Foundation

internal enum MLXGenerationDiagnosticEvent: Sendable, Equatable {
    case parameters(MLXGenerationParameterSnapshot)
    case promptCachePlan(MLXPromptCachePlanSnapshot)
    case speculativeDecoding(MLXSpeculativeDecodingSnapshot)
    case specPrefillPlan(MLXSpecPrefillPlanSnapshot)
    case dFlashPlan(MLXDFlashPlanSnapshot)
    case adaptivePrefillChunk(MLXAdaptivePrefillChunkSnapshot)
    case prefillChunk(MLXPrefillChunkSnapshot)
    case cacheSnapshot(MLXCacheSnapshot)
    case quantizedKVConversion(MLXQuantizedKVConversionSnapshot)
    case grammarConstraint(MLXGrammarConstraintSnapshot)
    case reasoningBudget(MLXReasoningBudgetSnapshot)
    case generatedToken(MLXGeneratedTokenSnapshot)
    case memoryGuard(MLXMemoryGuardSnapshot)
    case executionPlan(MLXGenerationExecutionPlanSnapshot)
    case admission(MLXGenerationAdmissionSnapshot)
    case batchRows(MLXGenerationBatchRowsSnapshot)
    case continuousBatchLogits(MLXContinuousBatchLogitsSnapshot)
    case pagedKVBlocks(MLXPagedKVBlockTableSnapshot)
    case persistentCacheInvalidation(MLXPersistentCacheInvalidationSnapshot)
    case promptCacheLookup(MLXPromptCacheLookupSnapshot)
    case promptCacheObservability(MLXPromptCacheObservabilitySnapshot)
    case sessionLifecycle(MLXSessionLifecycleSnapshot)
}

internal struct MLXGenerationParameterSnapshot: Sendable, Equatable {
    let maxTokens: Int?
    let maxKVSize: Int?
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let quantizedKVSkipLastLayer: Bool
    let prefillStepSize: Int
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
    let typicalP: Float
    let topNSigma: Float?
    let xtcProbability: Float
    let xtcThreshold: Float
    let xtcMinKeep: Int
    let xtcProtectedTokenCount: Int
    let mirostatVersion: MirostatSamplingVersion?
    let mirostatTau: Float?
    let mirostatEta: Float?
    let mirostatLearningTokens: Int32?
    let dryMultiplier: Float?
    let dryBase: Float?
    let dryAllowedLength: Int32?
    let dryPenaltyLastTokens: Int32?
    let drySequenceBreakerCount: Int
    let adaptivePTarget: Float?
    let adaptivePDecay: Float?
    let repetitionPenalty: Float?
    let repetitionContextSize: Int
    let presencePenalty: Float?
    let presenceContextSize: Int
    let frequencyPenalty: Float?
    let frequencyContextSize: Int
    let seed: Int?
    let logitBiasCount: Int
    let reasoningBudgetTokens: Int?
    let reasoningEndTokenCount: Int
    let suppressTokenCount: Int
    let grammarKind: GrammarConstraintKind?
}

internal struct MLXPromptCachePlanSnapshot: Sendable, Equatable {
    let promptTokenCount: Int
    let reusedTokenCount: Int
}

internal struct MLXPromptCacheLookupSnapshot: Sendable, Equatable {
    enum Strategy: String, Sendable {
        case linear
        case blockIndex
        case persistentSegments
        case persistentSnapshot
    }

    let strategy: Strategy
    let blockSize: Int?
    let matchedBlockCount: Int
    let candidateCount: Int
    let selectedIndex: Int?
    let reusedTokenCount: Int
}

internal struct MLXSpeculativeDecodingSnapshot: Sendable, Equatable {
    let numDraftTokens: Int
}

internal struct MLXSpecPrefillPlanSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case planned
        case skippedBelowThreshold
        case skippedDisabled
        case skippedImportanceMismatch
        case skippedNoReduction
        case skippedRuntimeUnavailable
    }

    let stage: Stage
    let promptTokenCount: Int
    let cachedTokenCount: Int
    let protectedPrefixTokenCount: Int
    let retainedTokenCount: Int
    let newPrefillTokenCount: Int
    let decodePositionOffset: Int
    let keepRate: Double?
    let thresholdTokens: Int?
    let chunkSize: Int?
    let message: String?
}

internal struct MLXDFlashPlanSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case planned
        case skippedDisabled
        case skippedMissingDraft
        case skippedContextTooLong
    }

    let stage: Stage
    let promptTokenCount: Int
    let draftModelID: String?
    let maxContextTokens: Int?
    let draftWindowSize: Int?
    let draftSinkSize: Int?
    let verifyMode: MLXDFlashVerifyMode?
    let usesMemoryCache: Bool
    let usesSSDCache: Bool
    let message: String?
}

internal struct MLXAdaptivePrefillChunkSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case disabled
        case profileUnavailable
        case limitUnavailable
        case unchanged
        case adjusted
    }

    let stage: Stage
    let requestedChunkSize: Int
    let selectedChunkSize: Int
    let minimumChunkSize: Int
    let promptTokenCount: Int
    let cachedTokenCount: Int
    let processedTokenCount: Int
    let currentMemoryBytes: Int64?
    let targetBytes: Int64?
    let predictedTransientBytes: Int64?
    let limitBytes: Int64?
    let limitSource: MLXMemoryGuardSnapshot.LimitSource?
    let observedBytesPerToken: Double?
    let observedSampleCount: Int
}

internal struct MLXPrefillChunkSnapshot: Sendable, Equatable {
    let chunkSize: Int
    let remainingTokenCount: Int
    let prefillStepSize: Int
    let memoryBeforeBytes: Int64?
    let memoryAfterBytes: Int64?
    let memoryDeltaBytes: Int64?
}

internal struct MLXCacheSnapshot: Sendable, Equatable {
    let label: String
    let entries: [MLXCacheEntrySnapshot]
}

internal struct MLXCacheEntrySnapshot: Sendable, Equatable {
    let typeName: String
    let offset: Int
    let maxSize: Int?
    let quantizedBits: Int?
    let quantizedGroupSize: Int?
}

internal struct MLXQuantizedKVConversionSnapshot: Sendable, Equatable {
    let offset: Int
    let kvBits: Int
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let quantizedKVSkipLastLayer: Bool
    let convertedCount: Int
}

internal struct MLXGrammarConstraintSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case compilerReady
        case compilerUnavailable
        case matcherPrepared
        case maskApplied
        case batchMaskApplied
        case mlxMaskPrepared
        case mlxMaskReused
        case maskSkipped
        case tokenAccepted
        case tokenRejected
        case processorFailedClosed
        case speculativeBypassed
    }

    let stage: Stage
    let kind: GrammarConstraintKind?
    let mode: GrammarTokenMask.Mode?
    let tokenCount: Int?
    let tokenID: Int?
    let vocabularySize: Int?
    let bitmaskSize: Int?
    let isCompleted: Bool?
    let isTerminated: Bool?
    let message: String?
}

internal struct MLXReasoningBudgetSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case tokenCounted
        case budgetReached
        case maskApplied
        case forcingEndMarker
        case forcedClosed
        case naturallyClosed
        case invalidEndToken
    }

    let stage: Stage
    let reasoningTokenCount: Int
    let forcedTokenID: Int?
    let message: String?
}

internal struct MLXGeneratedTokenSnapshot: Sendable, Equatable {
    let tokenID: Int
    let tokenText: String
    let index: Int
}

internal struct MLXMemoryGuardSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case disabled
        case profileUnavailable
        case modelLoadEstimateUnavailable
        case modelLoadAllowed
        case modelLoadRejected
        case limitUnavailable
        case allowed
        case rejected
    }

    enum LimitSource: String, Sendable {
        case processMemoryBudget
        case customLimit
        case metalRecommendedWorkingSet
        case hostAvailableMemory
    }

    let stage: Stage
    let tier: MLXMemoryGuardTier
    let promptTokenCount: Int
    let cachedTokenCount: Int
    let newTokenCount: Int
    let maximumGeneratedTokenCount: Int
    let prefillStepSize: Int
    let currentMemoryBytes: Int64?
    let estimatedPeakBytes: Int64?
    let limitBytes: Int64?
    let limitSource: LimitSource?
    let message: String?
}

internal struct MLXGenerationExecutionPlanSnapshot: Sendable, Equatable {
    let requestedStrategy: MLXGenerationExecutionStrategy
    let selectedStrategy: MLXGenerationExecutionStrategy
    let reason: MLXGenerationExecutionPlanReason
    let requestedMaxConcurrentRequests: Int
    let requestedMaxBatchSize: Int
    let effectiveMaxConcurrentRequests: Int
    let effectiveMaxBatchSize: Int
    let supportsContinuousBatching: Bool
}

internal struct MLXGenerationAdmissionSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case admitted
        case queued
        case released
        case queueFull
        case pauseUpdated
        case configured
        case batchAdmitted
    }

    let stage: Stage
    let activeCount: Int
    let waitingCount: Int
    let maxConcurrentRequests: Int
    let maxQueuedRequests: Int
    let maxBatchSize: Int
    let admittedCount: Int
    let admissionPaused: Bool
}

internal struct MLXGenerationBatchRowsSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case appended
        case removed
        case kept
        case updated
    }

    let stage: Stage
    let rowCount: Int
    let rowIDs: [Int]
    let affectedRowIDs: [Int]
}

internal struct MLXContinuousBatchLogitsSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case sampled
    }

    let stage: Stage
    let rowCount: Int
    let rowIDs: [Int]
    let tokenIDs: [Int]
}

internal struct MLXPagedKVBlockTableSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case allocated
        case attached
        case cleared
        case detached
        case evicted
        case forked
        case released
        case retained
        case touched
        case updated
    }

    let stage: Stage
    let capacity: Int
    let usedCount: Int
    let freeCount: Int
    let evictableCount: Int
    let blockIDs: [Int]
    let affectedBlockIDs: [Int]
    let refCounts: [Int]
    let rowID: Int?
}

internal struct MLXPersistentCacheInvalidationSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case staleSignatureSweep
        case skippedMissingIdentity
    }

    let stage: Stage
    let candidateCount: Int
    let removedCount: Int
    let payloadKinds: [String]
}

internal struct MLXSessionLifecycleSnapshot: Sendable, Equatable {
    enum Stage: String, Sendable {
        case generationStarted
        case generationFinished
        case unloadAdmissionPaused
        case unloadDeferred
        case unloadStarted
        case unloadFinished
        case unloadSkipped
        case unloadAdmissionResumed
    }

    let stage: Stage
    let activeGenerationCount: Int
    let pendingUnloadAfterGeneration: Bool
    let hasModelContainer: Bool
    let message: String?
}

internal enum MLXGenerationDiagnostics {
    private static let store = MLXGenerationDiagnosticStore()
    private static let promptCacheObservability = MLXPromptCacheObservabilityTracker()
    @TaskLocal private static var currentRunID: UUID?
    @TaskLocal private static var adaptivePrefillController: MLXAdaptivePrefillChunkController?

    internal static func withRecording<T>(
        _ operation: () async throws -> T
    ) async throws -> (result: T, events: [MLXGenerationDiagnosticEvent]) {
        let runID = UUID()
        store.reset(runID: runID)
        do {
            return try await $currentRunID.withValue(runID) {
                let result = try await operation()
                let events = store.events(runID: runID)
                store.reset(runID: runID)
                return (result, events)
            }
        } catch {
            store.reset(runID: runID)
            throw error
        }
    }

    internal static func reset() {
        store.reset(runID: currentRunID)
    }

    internal static func events() -> [MLXGenerationDiagnosticEvent] {
        store.events(runID: currentRunID)
    }

    internal static var currentAdaptivePrefillController: MLXAdaptivePrefillChunkController? {
        adaptivePrefillController
    }

    internal static func withAdaptivePrefillController<T>(
        _ controller: MLXAdaptivePrefillChunkController?,
        operation: () throws -> T
    ) rethrows -> T {
        try $adaptivePrefillController.withValue(controller) {
            try operation()
        }
    }

    internal static func resetPromptCacheObservability() {
        promptCacheObservability.reset()
    }

    internal static func promptCacheObservabilitySnapshot() -> MLXPromptCacheObservabilitySnapshot {
        promptCacheObservability.snapshot()
    }

    internal static func recordParameters(_ parameters: GenerateParameters) {
        store.record(.parameters(MLXGenerationParameterSnapshot(
            maxTokens: parameters.maxTokens,
            maxKVSize: parameters.maxKVSize,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart,
            quantizedKVSkipLastLayer: parameters.quantizedKVSkipLastLayer,
            prefillStepSize: parameters.prefillStepSize,
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            minP: parameters.minP,
            typicalP: parameters.typicalP,
            topNSigma: parameters.topNSigma,
            xtcProbability: parameters.xtcProbability,
            xtcThreshold: parameters.xtcThreshold,
            xtcMinKeep: parameters.xtcMinKeep,
            xtcProtectedTokenCount: parameters.xtcProtectedTokenIds.count,
            mirostatVersion: parameters.mirostat?.version,
            mirostatTau: parameters.mirostat?.tau,
            mirostatEta: parameters.mirostat?.eta,
            mirostatLearningTokens: parameters.mirostat?.learningTokens,
            dryMultiplier: parameters.dry?.multiplier,
            dryBase: parameters.dry?.base,
            dryAllowedLength: parameters.dry?.allowedLength,
            dryPenaltyLastTokens: parameters.dry?.penaltyLastTokens,
            drySequenceBreakerCount: parameters.drySequenceBreakerTokenIds.count,
            adaptivePTarget: parameters.adaptiveP?.target,
            adaptivePDecay: parameters.adaptiveP?.decay,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize,
            presencePenalty: parameters.presencePenalty,
            presenceContextSize: parameters.presenceContextSize,
            frequencyPenalty: parameters.frequencyPenalty,
            frequencyContextSize: parameters.frequencyContextSize,
            seed: parameters.seed,
            logitBiasCount: parameters.logitBias.count,
            reasoningBudgetTokens: parameters.reasoningBudgetTokens,
            reasoningEndTokenCount: parameters.reasoningEndTokenIds.count,
            suppressTokenCount: parameters.suppressTokenIds.count,
            grammarKind: parameters.grammar?.kind
        )), runID: currentRunID)
    }

    internal static func recordReasoningBudget(_ snapshot: MLXReasoningBudgetSnapshot) {
        store.record(.reasoningBudget(snapshot), runID: currentRunID)
    }

    internal static func recordPromptCachePlan(promptTokenCount: Int, reusedTokenCount: Int) {
        let observabilitySnapshot = promptCacheObservability.recordPlan(
            promptTokenCount: promptTokenCount,
            reusedTokenCount: reusedTokenCount
        )
        store.record(.promptCachePlan(MLXPromptCachePlanSnapshot(
            promptTokenCount: promptTokenCount,
            reusedTokenCount: reusedTokenCount
        )), runID: currentRunID)
        recordPromptCacheObservability(observabilitySnapshot)
    }

    internal static func recordPromptCacheLookup(_ snapshot: MLXPromptCacheLookupSnapshot) {
        store.record(.promptCacheLookup(snapshot), runID: currentRunID)
    }

    internal static func recordPromptCacheSSDSave() {
        recordPromptCacheObservability(promptCacheObservability.recordSSDSave())
    }

    internal static func recordPromptCacheSSDDiskLoad() {
        recordPromptCacheObservability(promptCacheObservability.recordSSDDiskLoad())
    }

    internal static func recordPromptCacheSSDHotHit() {
        recordPromptCacheObservability(promptCacheObservability.recordSSDHotHit())
    }

    internal static func recordPromptCacheHotCacheEviction() {
        recordPromptCacheObservability(promptCacheObservability.recordHotCacheEviction())
    }

    internal static func recordPromptCacheHotCachePromotion() {
        recordPromptCacheObservability(promptCacheObservability.recordHotCachePromotion())
    }

    internal static func recordPromptCacheEviction() {
        recordPromptCacheObservability(promptCacheObservability.recordEviction())
    }

    internal static func recordSpeculativeDecoding(numDraftTokens: Int) {
        store.record(.speculativeDecoding(MLXSpeculativeDecodingSnapshot(
            numDraftTokens: numDraftTokens
        )), runID: currentRunID)
    }

    internal static func recordSpecPrefillPlan(_ snapshot: MLXSpecPrefillPlanSnapshot) {
        store.record(.specPrefillPlan(snapshot), runID: currentRunID)
    }

    internal static func recordAdaptivePrefillChunk(_ snapshot: MLXAdaptivePrefillChunkSnapshot) {
        store.record(.adaptivePrefillChunk(snapshot), runID: currentRunID)
    }

    internal static func recordDFlashPlan(_ snapshot: MLXDFlashPlanSnapshot) {
        store.record(.dFlashPlan(snapshot), runID: currentRunID)
    }

    internal static func recordPrefillChunk(
        chunkSize: Int,
        remainingTokenCount: Int,
        prefillStepSize: Int,
        memoryBeforeBytes: Int64? = nil,
        memoryAfterBytes: Int64? = nil
    ) {
        let memoryDeltaBytes = memoryBeforeBytes.flatMap { before in
            memoryAfterBytes.map { after in
                after - before
            }
        }
        store.record(.prefillChunk(MLXPrefillChunkSnapshot(
            chunkSize: chunkSize,
            remainingTokenCount: remainingTokenCount,
            prefillStepSize: prefillStepSize,
            memoryBeforeBytes: memoryBeforeBytes,
            memoryAfterBytes: memoryAfterBytes,
            memoryDeltaBytes: memoryDeltaBytes
        )), runID: currentRunID)
    }

    internal static func recordCacheSnapshot(label: String, cache: [KVCache]) {
        let entries = cache.map { entry in
            MLXCacheEntrySnapshot(
                typeName: String(describing: type(of: entry)),
                offset: entry.offset,
                maxSize: entry.maxSize,
                quantizedBits: (entry as? QuantizedKVCacheProtocol)?.bits,
                quantizedGroupSize: (entry as? QuantizedKVCacheProtocol)?.groupSize
            )
        }
        store.record(
            .cacheSnapshot(MLXCacheSnapshot(label: label, entries: entries)),
            runID: currentRunID
        )
    }

    internal static func recordQuantizedKVConversion(
        offset: Int,
        kvBits: Int,
        kvGroupSize: Int,
        quantizedKVStart: Int,
        quantizedKVSkipLastLayer: Bool,
        convertedCount: Int
    ) {
        store.record(.quantizedKVConversion(MLXQuantizedKVConversionSnapshot(
            offset: offset,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart,
            quantizedKVSkipLastLayer: quantizedKVSkipLastLayer,
            convertedCount: convertedCount
        )), runID: currentRunID)
    }

    internal static func recordGrammarConstraint(_ snapshot: MLXGrammarConstraintSnapshot) {
        store.record(.grammarConstraint(snapshot), runID: currentRunID)
    }

    internal static func recordGeneratedToken(tokenID: Int, tokenText: String, index: Int) {
        guard let currentRunID else {
            return
        }
        store.record(.generatedToken(MLXGeneratedTokenSnapshot(
            tokenID: tokenID,
            tokenText: tokenText,
            index: index
        )), runID: currentRunID)
    }

    internal static func recordMemoryGuard(_ snapshot: MLXMemoryGuardSnapshot) {
        store.record(.memoryGuard(snapshot), runID: currentRunID)
    }

    internal static func recordExecutionPlan(_ plan: MLXGenerationExecutionPlan) {
        store.record(.executionPlan(MLXGenerationExecutionPlanSnapshot(
            requestedStrategy: plan.requestedStrategy,
            selectedStrategy: plan.selectedStrategy,
            reason: plan.reason,
            requestedMaxConcurrentRequests: plan.requestedScheduling.maxConcurrentRequests,
            requestedMaxBatchSize: plan.requestedScheduling.maxBatchSize,
            effectiveMaxConcurrentRequests: plan.effectiveScheduling.maxConcurrentRequests,
            effectiveMaxBatchSize: plan.effectiveScheduling.maxBatchSize,
            supportsContinuousBatching: plan.capabilities.supportsContinuousBatching
        )), runID: currentRunID)
    }

    internal static func recordAdmission(_ snapshot: MLXGenerationAdmissionSnapshot) {
        store.record(.admission(snapshot), runID: currentRunID)
    }

    internal static func recordBatchRows(_ snapshot: MLXGenerationBatchRowsSnapshot) {
        store.record(.batchRows(snapshot), runID: currentRunID)
    }

    internal static func recordContinuousBatchLogits(
        _ snapshot: MLXContinuousBatchLogitsSnapshot
    ) {
        store.record(.continuousBatchLogits(snapshot), runID: currentRunID)
    }

    internal static func recordPagedKVBlocks(_ snapshot: MLXPagedKVBlockTableSnapshot) {
        store.record(.pagedKVBlocks(snapshot), runID: currentRunID)
    }

    internal static func recordPersistentCacheInvalidation(
        _ snapshot: MLXPersistentCacheInvalidationSnapshot
    ) {
        store.record(.persistentCacheInvalidation(snapshot), runID: currentRunID)
    }

    internal static func recordSessionLifecycle(_ snapshot: MLXSessionLifecycleSnapshot) {
        store.record(.sessionLifecycle(snapshot), runID: currentRunID)
    }

    private static func recordPromptCacheObservability(
        _ snapshot: MLXPromptCacheObservabilitySnapshot
    ) {
        store.record(.promptCacheObservability(snapshot), runID: currentRunID)
    }
}

private final class MLXGenerationDiagnosticStore: @unchecked Sendable {
    private let lock = NSLock()
    private var defaultEvents: [MLXGenerationDiagnosticEvent] = []
    private var keyedEvents: [UUID: [MLXGenerationDiagnosticEvent]] = [:]

    func reset(runID: UUID?) {
        lock.lock()
        if let runID {
            keyedEvents[runID] = []
        } else {
            defaultEvents.removeAll(keepingCapacity: true)
            keyedEvents.removeAll(keepingCapacity: true)
        }
        lock.unlock()
    }

    func events(runID: UUID?) -> [MLXGenerationDiagnosticEvent] {
        lock.lock()
        let snapshot = runID.map { keyedEvents[$0] ?? [] } ?? defaultEvents
        lock.unlock()
        return snapshot
    }

    func record(_ event: MLXGenerationDiagnosticEvent, runID: UUID?) {
        lock.lock()
        if let runID {
            keyedEvents[runID, default: []].append(event)
        } else {
            defaultEvents.append(event)
        }
        lock.unlock()
    }
}

/// Metrics data collected during text generation
internal struct MetricsData {
    let generationStartTime: ContinuousClock.Instant
    let promptStartTime: ContinuousClock.Instant
    let promptEndTime: ContinuousClock.Instant
    let firstTokenTime: ContinuousClock.Instant?
    let promptTokenCount: Int
    let generatedTokenCount: Int
    let kvCacheBytes: Int64?
    let kvCacheEntries: Int?
    let promptCacheReusedTokenCount: Int?
    let stopReason: GenerationMetrics.StopReason
    let parameters: GenerateParameters
}

/// Context for token processing operations
internal struct TokenContext {
    let state: GenerationState
    let context: ModelContext
    let input: LLMInput
    let continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    let clock: ContinuousClock
}

/// Context for generation operations
internal struct GenerationContext {
    let modelContext: ModelContext
    let input: LLMInput
    let parameters: GenerateParameters
    let generationStartTime: ContinuousClock.Instant
    let continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation
    let clock: ContinuousClock
    let runtimePreferences: ModelRuntimePreferences
    let memoryProfile: MLXModelMemoryProfile?
}
