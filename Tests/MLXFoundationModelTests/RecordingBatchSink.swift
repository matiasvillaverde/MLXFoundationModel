import Foundation
@testable import MLXLocalModels

final class RecordingBatchSink: @unchecked Sendable {
    private let lock = NSLock()
    private var chunks: [LLMStreamChunk] = []
    private var failures: [any Error] = []
    private var reasons: [MLXContinuousBatchFinishReason?] = []

    deinit {
        // Required by the strict lint profile.
    }

    func streamSink() -> MLXContinuousBatchStreamSink {
        MLXContinuousBatchStreamSink(
            yield: { self.append($0) },
            finish: { self.append(reason: $0) },
            fail: { self.append(error: $0) }
        )
    }

    func texts() -> [String] {
        withLock { chunks.map(\.text) }
    }

    func tokenIDs() -> [[Int]] {
        withLock { chunks.map(\.tokenIDs) }
    }

    func finishReasons() -> [MLXContinuousBatchFinishReason] {
        withLock { reasons.compactMap(\.self) }
    }

    func failureCount() -> Int {
        withLock { failures.count }
    }

    private func append(_ chunk: LLMStreamChunk) {
        lock.lock()
        chunks.append(chunk)
        lock.unlock()
    }

    private func append(reason: MLXContinuousBatchFinishReason?) {
        lock.lock()
        reasons.append(reason)
        lock.unlock()
    }

    private func append(error: any Error) {
        lock.lock()
        failures.append(error)
        lock.unlock()
    }

    private func withLock<T>(_ action: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return action()
    }
}
