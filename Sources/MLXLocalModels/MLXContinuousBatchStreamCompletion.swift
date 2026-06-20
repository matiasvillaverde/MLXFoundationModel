import Foundation

internal final class MLXContinuousBatchStreamCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private let state: GenerationState
    private var continuation: CheckedContinuation<Void, any Error>?
    private var result: Result<Void, any Error>?

    internal init(state: GenerationState) {
        self.state = state
        lock.name = "org.mlxfoundationmodel.continuous-batch-completion"
    }

    deinit {
        complete(.failure(CancellationError()))
    }

    internal func streamSink(
        continuation: AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation
    ) -> MLXContinuousBatchStreamSink {
        MLXContinuousBatchStreamSink(
            yield: { continuation.yield($0) },
            finish: { self.finish(reason: $0) },
            fail: { self.fail($0) }
        )
    }

    internal func wait() async throws {
        try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            if let result {
                lock.unlock()
                continuation.resume(with: result)
                return
            }
            self.continuation = continuation
            lock.unlock()
        }
    }

    internal func cancel() {
        state.stopReason = .userRequested
        complete(.failure(CancellationError()))
    }

    private func finish(reason: MLXContinuousBatchFinishReason?) {
        state.stopReason = reason?.stopReason ?? state.stopReason
        complete(.success(()))
    }

    private func fail(_ error: any Error) {
        state.stopReason = .error
        complete(.failure(error))
    }

    private func complete(_ result: Result<Void, any Error>) {
        lock.lock()
        guard self.result == nil else {
            lock.unlock()
            return
        }
        self.result = result
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume(with: result)
    }
}

extension MLXContinuousBatchFinishReason {
    var stopReason: GenerationMetrics.StopReason {
        switch self {
        case .maximumTokenCount:
            .maxTokens

        case .stopToken:
            .endOfSequence

        case .streamRequestedStop:
            .stopSequence
        }
    }
}
