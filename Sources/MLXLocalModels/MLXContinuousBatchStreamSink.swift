internal struct MLXContinuousBatchStreamSink: Sendable {
    internal let fail: @Sendable (any Error) -> Void
    internal let finish: @Sendable (MLXContinuousBatchFinishReason?) -> Void
    internal let yield: @Sendable (LLMStreamChunk) -> Void

    internal init(
        yield: @escaping @Sendable (LLMStreamChunk) -> Void,
        finish: @escaping @Sendable (MLXContinuousBatchFinishReason?) -> Void,
        fail: @escaping @Sendable (any Error) -> Void
    ) {
        self.fail = fail
        self.finish = finish
        self.yield = yield
    }

    internal static func stream(
        _ continuation: AsyncThrowingStream<LLMStreamChunk, any Error>.Continuation
    ) -> Self {
        Self(
            yield: { continuation.yield($0) },
            finish: { _ in continuation.finish() },
            fail: { continuation.finish(throwing: $0) }
        )
    }
}
