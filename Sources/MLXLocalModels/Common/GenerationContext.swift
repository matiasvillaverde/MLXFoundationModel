import Foundation

internal enum MLXGenerationDiagnosticEvent: Sendable, Equatable {
    case parameters(MLXGenerationParameterSnapshot)
    case promptCachePlan(MLXPromptCachePlanSnapshot)
    case speculativeDecoding(MLXSpeculativeDecodingSnapshot)
    case prefillChunk(MLXPrefillChunkSnapshot)
    case cacheSnapshot(MLXCacheSnapshot)
    case quantizedKVConversion(MLXQuantizedKVConversionSnapshot)
}

internal struct MLXGenerationParameterSnapshot: Sendable, Equatable {
    let maxTokens: Int?
    let maxKVSize: Int?
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let prefillStepSize: Int
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
    let repetitionPenalty: Float?
    let repetitionContextSize: Int
    let presencePenalty: Float?
    let presenceContextSize: Int
    let frequencyPenalty: Float?
    let frequencyContextSize: Int
    let seed: Int?
    let logitBiasCount: Int
}

internal struct MLXPromptCachePlanSnapshot: Sendable, Equatable {
    let promptTokenCount: Int
    let reusedTokenCount: Int
}

internal struct MLXSpeculativeDecodingSnapshot: Sendable, Equatable {
    let numDraftTokens: Int
}

internal struct MLXPrefillChunkSnapshot: Sendable, Equatable {
    let chunkSize: Int
    let remainingTokenCount: Int
    let prefillStepSize: Int
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
    let convertedCount: Int
}

internal enum MLXGenerationDiagnostics {
    private static let store = MLXGenerationDiagnosticStore()
    @TaskLocal private static var currentRunID: UUID?

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

    internal static func recordParameters(_ parameters: GenerateParameters) {
        store.record(.parameters(MLXGenerationParameterSnapshot(
            maxTokens: parameters.maxTokens,
            maxKVSize: parameters.maxKVSize,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart,
            prefillStepSize: parameters.prefillStepSize,
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            minP: parameters.minP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize,
            presencePenalty: parameters.presencePenalty,
            presenceContextSize: parameters.presenceContextSize,
            frequencyPenalty: parameters.frequencyPenalty,
            frequencyContextSize: parameters.frequencyContextSize,
            seed: parameters.seed,
            logitBiasCount: parameters.logitBias.count
        )), runID: currentRunID)
    }

    internal static func recordPromptCachePlan(promptTokenCount: Int, reusedTokenCount: Int) {
        store.record(.promptCachePlan(MLXPromptCachePlanSnapshot(
            promptTokenCount: promptTokenCount,
            reusedTokenCount: reusedTokenCount
        )), runID: currentRunID)
    }

    internal static func recordSpeculativeDecoding(numDraftTokens: Int) {
        store.record(.speculativeDecoding(MLXSpeculativeDecodingSnapshot(
            numDraftTokens: numDraftTokens
        )), runID: currentRunID)
    }

    internal static func recordPrefillChunk(
        chunkSize: Int,
        remainingTokenCount: Int,
        prefillStepSize: Int
    ) {
        store.record(.prefillChunk(MLXPrefillChunkSnapshot(
            chunkSize: chunkSize,
            remainingTokenCount: remainingTokenCount,
            prefillStepSize: prefillStepSize
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
        convertedCount: Int
    ) {
        store.record(.quantizedKVConversion(MLXQuantizedKVConversionSnapshot(
            offset: offset,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart,
            convertedCount: convertedCount
        )), runID: currentRunID)
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
}
