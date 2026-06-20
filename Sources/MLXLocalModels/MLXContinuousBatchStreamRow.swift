internal struct MLXContinuousBatchStreamRow: Sendable {
    internal let id: MLXGenerationBatchRowID
    private let handleToken: @Sendable (
        Int,
        MLXContinuousBatchStreamSink
    ) -> MLXContinuousBatchStreamTokenDisposition
    private let sink: MLXContinuousBatchStreamSink

    internal init(
        id: MLXGenerationBatchRowID,
        tokenText: @escaping @Sendable (Int) -> String,
        sink: MLXContinuousBatchStreamSink
    ) {
        self.id = id
        self.sink = sink
        self.handleToken = { tokenID, sink in
            let text = tokenText(tokenID)
            guard !text.isEmpty else {
                return .suppressed
            }
            sink.yield(LLMStreamChunk(
                text: text,
                event: .text,
                tokenCount: 1,
                tokenIDs: [tokenID]
            ))
            return .streamed
        }
    }

    internal init(
        id: MLXGenerationBatchRowID,
        sink: MLXContinuousBatchStreamSink,
        handleToken: @escaping @Sendable (
            Int,
            MLXContinuousBatchStreamSink
        ) -> MLXContinuousBatchStreamTokenDisposition
    ) {
        self.id = id
        self.sink = sink
        self.handleToken = handleToken
    }

    internal func finish(reason: MLXContinuousBatchFinishReason?) {
        sink.finish(reason)
    }

    internal func fail(_ error: any Error) {
        sink.fail(error)
    }

    internal func stream(tokenID: Int) -> MLXContinuousBatchStreamTokenDisposition {
        handleToken(tokenID, sink)
    }
}
