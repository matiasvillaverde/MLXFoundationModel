import Foundation

internal final class MLXContinuousBatchLifecycleEmitter: @unchecked Sendable {
    private let cachedTokenCount: Int
    private let lock = NSLock()
    private let promptTokenCount: Int
    private var didEmitDecodeStart = false
    private var didEmitPromptEnd = false

    internal init(
        promptTokenCount: Int,
        cachedTokenCount: Int
    ) {
        self.promptTokenCount = promptTokenCount
        self.cachedTokenCount = cachedTokenCount
        lock.name = "org.mlxfoundationmodel.continuous-batch-lifecycle"
    }

    deinit {
        // Required by the local lint profile for reference types.
    }

    internal func emitPromptEndIfNeeded(
        to yield: @Sendable (LLMStreamChunk) -> Void
    ) {
        if takePromptEnd() {
            yield(.lifecycle(.promptProcessingEnded(
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount
            )))
        }
    }

    internal func emitDecodeStartIfNeeded(
        to yield: @Sendable (LLMStreamChunk) -> Void
    ) {
        emitPromptEndIfNeeded(to: yield)
        if takeDecodeStart() {
            yield(.lifecycle(.init(phase: .decode, state: .started)))
        }
    }

    private func takePromptEnd() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !didEmitPromptEnd else {
            return false
        }
        didEmitPromptEnd = true
        return true
    }

    private func takeDecodeStart() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !didEmitDecodeStart else {
            return false
        }
        didEmitDecodeStart = true
        return true
    }
}

extension LLMStreamChunk {
    internal static func lifecycle(_ event: StreamLifecycleEvent) -> Self {
        Self(text: "", event: .lifecycle(event))
    }
}

extension AsyncThrowingStream<LLMStreamChunk, Error>.Continuation {
    internal func yieldLifecycle(_ event: StreamLifecycleEvent) {
        yield(.lifecycle(event))
    }
}

extension StreamLifecycleEvent {
    internal static func promptProcessingProgress(
        promptTokenCount: Int,
        cachedTokenCount: Int
    ) -> Self {
        let completedTokenCount = clampedCachedTokenCount(
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount
        )
        return Self(
            phase: .promptProcessing,
            state: .progress,
            completedUnitCount: Int64(completedTokenCount),
            totalUnitCount: Int64(max(0, promptTokenCount)),
            cachedUnitCount: Int64(completedTokenCount),
            message: "prompt-planned"
        )
    }

    internal static func promptProcessingEnded(
        promptTokenCount: Int,
        cachedTokenCount: Int
    ) -> Self {
        let totalTokenCount = max(0, promptTokenCount)
        return Self(
            phase: .promptProcessing,
            state: .ended,
            completedUnitCount: Int64(totalTokenCount),
            totalUnitCount: Int64(totalTokenCount),
            cachedUnitCount: Int64(clampedCachedTokenCount(
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount
            ))
        )
    }

    internal init(modelLoadProgress progress: Progress, message: String?) {
        let progressMessage = progress.localizedDescription.isEmpty
            ? nil
            : progress.localizedDescription
        self.init(
            phase: .modelLoad,
            state: .progress,
            completedUnitCount: progress.completedUnitCount,
            totalUnitCount: progress.totalUnitCount >= 0 ? progress.totalUnitCount : nil,
            message: message ?? progressMessage
        )
    }

    private static func clampedCachedTokenCount(
        promptTokenCount: Int,
        cachedTokenCount: Int
    ) -> Int {
        min(max(0, cachedTokenCount), max(0, promptTokenCount))
    }
}

extension MLXContinuousBatchStreamSink {
    internal func reportingLifecycle(
        promptTokenCount: Int,
        cachedTokenCount: Int
    ) -> Self {
        let emitter = MLXContinuousBatchLifecycleEmitter(
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount
        )
        return Self(
            yield: { chunk in
                emitter.emitDecodeStartIfNeeded(to: yield)
                yield(chunk)
            },
            finish: { reason in
                emitter.emitPromptEndIfNeeded(to: yield)
                finish(reason)
            },
            fail: fail
        )
    }
}
