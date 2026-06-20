import Foundation
import MLXLocalModels

final class LLMChunkStreamSource: @unchecked Sendable {
    private typealias Continuation = AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation

    private let lock = NSLock()
    private var continuation: Continuation?

    deinit {
        // Required by the package lint profile.
    }

    func stream() -> AsyncThrowingStream<LLMStreamChunk, any Error> {
        AsyncThrowingStream { continuation in
            lock.lock()
            self.continuation = continuation
            lock.unlock()
        }
    }

    func yield(_ chunk: LLMStreamChunk) {
        currentContinuation?.yield(chunk)
    }

    func finish() {
        currentContinuation?.finish()
    }

    private var currentContinuation: Continuation? {
        lock.lock()
        defer { lock.unlock() }
        return continuation
    }
}
