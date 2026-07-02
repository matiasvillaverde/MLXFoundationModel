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
            yield(.lifecycle(.init(
                phase: .promptProcessing,
                state: .ended,
                completedUnitCount: Int64(promptTokenCount),
                totalUnitCount: Int64(promptTokenCount),
                cachedUnitCount: Int64(cachedTokenCount)
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
