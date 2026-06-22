import Foundation
import MLXLocalModels

final class LLMChunkStreamSource: @unchecked Sendable {
    private typealias Continuation = AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation

    private let lock = NSLock()
    private var continuation: Continuation?
    private var pendingChunks: [LLMStreamChunk] = []
    private var didFinish = false

    deinit {
        // Required by the package lint profile.
    }

    func stream() -> AsyncThrowingStream<LLMStreamChunk, any Error> {
        AsyncThrowingStream { continuation in
            lock.lock()
            self.continuation = continuation
            let chunks = pendingChunks
            let shouldFinish = didFinish
            pendingChunks = []
            lock.unlock()
            for chunk in chunks {
                continuation.yield(chunk)
            }
            if shouldFinish {
                continuation.finish()
            }
        }
    }

    func yield(_ chunk: LLMStreamChunk) {
        lock.lock()
        guard let continuation else {
            pendingChunks.append(chunk)
            lock.unlock()
            return
        }
        lock.unlock()
        continuation.yield(chunk)
    }

    func finish() {
        lock.lock()
        didFinish = true
        let continuation = continuation
        lock.unlock()
        continuation?.finish()
    }
}
