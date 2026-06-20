import Foundation
import MLX

internal struct MLXAdaptivePrefillObservation: Equatable, Sendable {
    let bytesPerToken: Double
    let sampleCount: Int
}

internal final class MLXAdaptivePrefillTransientTracker: @unchecked Sendable {
    private static let alpha = 0.3
    private let lock = NSLock()
    private var ewmaBytesPerToken = 0.0
    private var sampleCount = 0

    func record(tokenCount: Int, transientBytes: Int64) {
        guard tokenCount > 0, transientBytes > 0 else {
            return
        }

        let bytesPerToken = Double(transientBytes) / Double(tokenCount)
        lock.withLock {
            if sampleCount == 0 {
                ewmaBytesPerToken = bytesPerToken
            } else {
                ewmaBytesPerToken = Self.alpha * bytesPerToken
                    + (1.0 - Self.alpha) * ewmaBytesPerToken
            }
            sampleCount += 1
        }
    }

    func prediction(tokenCount: Int, safetyFactor: Double = 1.2) -> Int64? {
        guard tokenCount > 0 else {
            return nil
        }
        let value = lock.withLock { () -> Double? in
            guard sampleCount > 0 else {
                return nil
            }
            return ewmaBytesPerToken
        }
        guard let value else {
            return nil
        }
        let predicted = value * Double(tokenCount) * safetyFactor
        guard predicted.isFinite, predicted > 0 else {
            return nil
        }
        return predicted >= Double(Int64.max) ? Int64.max : Int64(predicted.rounded(.up))
    }

    var observation: MLXAdaptivePrefillObservation? {
        lock.withLock {
            guard sampleCount > 0 else {
                return nil
            }
            return MLXAdaptivePrefillObservation(
                bytesPerToken: ewmaBytesPerToken,
                sampleCount: sampleCount
            )
        }
    }
}

internal final class MLXAdaptivePrefillChunkController: @unchecked Sendable {
    private let configuration: MLXMemoryGuardConfiguration
    private let profile: MLXModelMemoryProfile
    private let promptTokenCount: Int
    private let initialCachedTokenCount: Int
    private let requestedChunkSize: Int
    private let minimumChunkSize: Int
    private let baselineMemorySnapshot: MLXRuntimeMemorySnapshot
    private let tracker: MLXAdaptivePrefillTransientTracker
    private let lock = NSLock()
    private var processedTokenCount = 0
    private var currentMemoryBytes: Int64

    init(
        configuration: MLXMemoryGuardConfiguration,
        profile: MLXModelMemoryProfile,
        promptTokenCount: Int,
        cachedTokenCount: Int,
        requestedChunkSize: Int,
        minimumChunkSize: Int = MLXAdaptivePrefillChunkSizer.defaultMinimumChunkSize,
        memorySnapshot: MLXRuntimeMemorySnapshot = .live(),
        tracker: MLXAdaptivePrefillTransientTracker = MLXAdaptivePrefillTransientTracker()
    ) {
        self.configuration = configuration
        self.profile = profile
        self.promptTokenCount = max(0, promptTokenCount)
        self.initialCachedTokenCount = max(0, cachedTokenCount)
        self.requestedChunkSize = max(1, requestedChunkSize)
        self.minimumChunkSize = max(1, minimumChunkSize)
        self.baselineMemorySnapshot = memorySnapshot
        self.currentMemoryBytes = memorySnapshot.currentMemoryBytes
        self.tracker = tracker
    }

    func decision(remainingTokenCount: Int) -> MLXAdaptivePrefillChunkSizer.Decision {
        let remaining = max(0, remainingTokenCount)
        let state = lock.withLock {
            (processedTokenCount, currentMemoryBytes)
        }
        let candidate = min(requestedChunkSize, max(remaining, 1))
        let observation = tracker.observation
        let predicted = tracker.prediction(tokenCount: candidate)
        return MLXAdaptivePrefillChunkSizer.decision(
            configuration: configuration,
            profile: profile,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: min(initialCachedTokenCount + state.0, promptTokenCount),
            processedTokenCount: state.0,
            requestedChunkSize: candidate,
            minimumChunkSize: min(minimumChunkSize, candidate),
            memorySnapshot: baselineMemorySnapshot.replacingCurrentMemoryBytes(state.1),
            measuredPredictedTransientBytes: predicted,
            observedBytesPerToken: observation?.bytesPerToken,
            observedSampleCount: observation?.sampleCount ?? 0
        )
    }

    func recordChunk(tokenCount: Int, memoryBeforeBytes: Int64, memoryAfterBytes: Int64) {
        guard tokenCount > 0 else {
            return
        }
        lock.withLock {
            processedTokenCount += tokenCount
            currentMemoryBytes = max(0, memoryAfterBytes)
        }
        tracker.record(
            tokenCount: tokenCount,
            transientBytes: memoryAfterBytes - memoryBeforeBytes
        )
    }
}

internal enum MLXAdaptivePrefillChunkSizer {
    internal struct Decision: Equatable, Sendable {
        let snapshot: MLXAdaptivePrefillChunkSnapshot

        var selectedChunkSize: Int {
            snapshot.selectedChunkSize
        }
    }

    private static let headroomSafety = 0.90
    private static let transientSafety = 1.30
    internal static let defaultMinimumChunkSize = 256

    static func decision(
        configuration: MLXMemoryGuardConfiguration,
        profile: MLXModelMemoryProfile?,
        promptTokenCount: Int,
        cachedTokenCount: Int,
        processedTokenCount: Int = 0,
        requestedChunkSize: Int,
        minimumChunkSize: Int = defaultMinimumChunkSize,
        memorySnapshot: MLXRuntimeMemorySnapshot,
        measuredPredictedTransientBytes: Int64? = nil,
        observedBytesPerToken: Double? = nil,
        observedSampleCount: Int = 0
    ) -> Decision {
        let current = memorySnapshot.currentMemoryBytes
            + (configuration.includeCacheMemory ? memorySnapshot.cacheMemoryBytes : 0)
        let limitBytes = MLXRuntimeMemoryGuard.limitBytes(
            configuration: configuration,
            memorySnapshot: memorySnapshot,
            currentMemoryBytes: current
        )
        return decision(
            configuration: configuration,
            profile: profile,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount,
            processedTokenCount: processedTokenCount,
            requestedChunkSize: requestedChunkSize,
            minimumChunkSize: minimumChunkSize,
            currentMemoryBytes: current,
            cacheMemoryBytes: 0,
            physicalMemoryBytes: memorySnapshot.physicalMemoryBytes,
            metalLimitBytes: memorySnapshot.metalLimitBytes,
            limitBytes: limitBytes,
            measuredPredictedTransientBytes: measuredPredictedTransientBytes,
            observedBytesPerToken: observedBytesPerToken,
            observedSampleCount: observedSampleCount
        )
    }

    static func decision(
        configuration: MLXMemoryGuardConfiguration,
        profile: MLXModelMemoryProfile?,
        promptTokenCount: Int,
        cachedTokenCount: Int,
        processedTokenCount: Int = 0,
        requestedChunkSize: Int,
        minimumChunkSize: Int = defaultMinimumChunkSize,
        currentMemoryBytes: Int64 = Int64(Memory.activeMemory),
        cacheMemoryBytes: Int64 = Int64(Memory.cacheMemory),
        physicalMemoryBytes: Int64 = Int64(ProcessInfo.processInfo.physicalMemory),
        metalLimitBytes: Int64? = nil,
        limitBytes: Int64? = nil,
        measuredPredictedTransientBytes: Int64? = nil,
        observedBytesPerToken: Double? = nil,
        observedSampleCount: Int = 0
    ) -> Decision {
        let requested = max(1, requestedChunkSize)
        let minimum = min(max(1, minimumChunkSize), requested)
        let promptCount = max(0, promptTokenCount)
        let cachedCount = min(max(0, cachedTokenCount), promptCount)
        let processedCount = min(max(0, processedTokenCount), promptCount)
        let current = max(0, currentMemoryBytes)
            + (configuration.includeCacheMemory ? max(0, cacheMemoryBytes) : 0)

        guard configuration.tier != .off else {
            return decision(
                stage: .disabled,
                requested: requested,
                selected: requested,
                minimum: minimum,
                promptCount: promptCount,
                cachedCount: cachedCount,
                processedCount: processedCount
            )
        }

        guard let profile else {
            return decision(
                stage: .profileUnavailable,
                requested: requested,
                selected: requested,
                minimum: minimum,
                promptCount: promptCount,
                cachedCount: cachedCount,
                processedCount: processedCount
            )
        }

        let resolvedLimitBytes = limitBytes ?? MLXRuntimeMemoryGuard.limitBytes(
            configuration: configuration,
            physicalMemoryBytes: physicalMemoryBytes,
            metalLimitBytes: metalLimitBytes
        )
        guard let resolvedLimitBytes else {
            return decision(
                stage: .limitUnavailable,
                requested: requested,
                selected: requested,
                minimum: minimum,
                promptCount: promptCount,
                cachedCount: cachedCount,
                processedCount: processedCount
            )
        }

        let target = max(1, Int64((Double(resolvedLimitBytes) * headroomSafety).rounded(.down)))
        let newTokenCount = max(promptCount - cachedCount, 0)
        let candidate = min(requested, max(newTokenCount, 1))
        let predicted = measuredPredictedTransientBytes ?? predictedTransientBytes(
            profile: profile,
            chunkTokenCount: candidate,
            kvLength: cachedCount
        )

        guard predicted > 0, current + predicted > target else {
            return decision(
                stage: .unchanged,
                requested: requested,
                selected: requested,
                minimum: minimum,
                promptCount: promptCount,
                cachedCount: cachedCount,
                processedCount: processedCount,
                current: current,
                target: target,
                predicted: predicted,
                limit: resolvedLimitBytes,
                observedBytesPerToken: observedBytesPerToken,
                observedSampleCount: observedSampleCount
            )
        }

        let perToken = Double(predicted) / Double(candidate)
        let headroom = max(target - current, 0)
        let fittingChunk = perToken > 0
            ? Int((Double(headroom) / perToken).rounded(.down))
            : requested
        let selected = max(minimum, min(requested, fittingChunk))
        return decision(
            stage: selected < requested ? .adjusted : .unchanged,
            requested: requested,
            selected: selected,
            minimum: minimum,
            promptCount: promptCount,
            cachedCount: cachedCount,
            processedCount: processedCount,
            current: current,
            target: target,
            predicted: predicted,
            limit: resolvedLimitBytes,
            observedBytesPerToken: observedBytesPerToken,
            observedSampleCount: observedSampleCount
        )
    }

    private static func predictedTransientBytes(
        profile: MLXModelMemoryProfile,
        chunkTokenCount: Int,
        kvLength: Int
    ) -> Int64 {
        guard chunkTokenCount > 0 else {
            return 0
        }
        let transient = profile.estimateSDPAActivationBytes(
            queryTokenCount: chunkTokenCount,
            kvLength: max(0, kvLength) + chunkTokenCount
        )
        let kvGrowth = profile.estimatePromptKVBytes(tokenCount: chunkTokenCount)
        let total = Double(transient + kvGrowth) * transientSafety
        guard total.isFinite, total > 0 else {
            return 0
        }
        return total >= Double(Int64.max) ? Int64.max : Int64(total.rounded(.up))
    }

    private static func decision(
        stage: MLXAdaptivePrefillChunkSnapshot.Stage,
        requested: Int,
        selected: Int,
        minimum: Int,
        promptCount: Int,
        cachedCount: Int,
        processedCount: Int,
        current: Int64? = nil,
        target: Int64? = nil,
        predicted: Int64? = nil,
        limit: Int64? = nil,
        observedBytesPerToken: Double? = nil,
        observedSampleCount: Int = 0
    ) -> Decision {
        Decision(snapshot: MLXAdaptivePrefillChunkSnapshot(
            stage: stage,
            requestedChunkSize: requested,
            selectedChunkSize: selected,
            minimumChunkSize: minimum,
            promptTokenCount: promptCount,
            cachedTokenCount: cachedCount,
            processedTokenCount: processedCount,
            currentMemoryBytes: current,
            targetBytes: target,
            predictedTransientBytes: predicted,
            limitBytes: limit,
            limitSource: nil,
            observedBytesPerToken: observedBytesPerToken,
            observedSampleCount: observedSampleCount
        ))
    }
}
