internal struct MLXContinuousBatchGenerationRequest: Sendable {
    internal let generationRow: MLXContinuousBatchGenerationRow
    internal let handleToken: @Sendable (
        Int,
        MLXContinuousBatchStreamSink
    ) -> MLXContinuousBatchStreamTokenDisposition
    internal let sink: MLXContinuousBatchStreamSink

    internal init(
        generationRow: MLXContinuousBatchGenerationRow,
        tokenText: @escaping @Sendable (Int) -> String,
        sink: MLXContinuousBatchStreamSink
    ) {
        self.sink = sink
        self.generationRow = generationRow
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
        generationRow: MLXContinuousBatchGenerationRow,
        sink: MLXContinuousBatchStreamSink,
        handleToken: @escaping @Sendable (
            Int,
            MLXContinuousBatchStreamSink
        ) -> MLXContinuousBatchStreamTokenDisposition
    ) {
        self.sink = sink
        self.generationRow = generationRow
        self.handleToken = handleToken
    }
}
