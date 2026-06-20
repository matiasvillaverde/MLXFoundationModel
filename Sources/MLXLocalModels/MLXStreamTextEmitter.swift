internal enum MLXStreamTextEmitter {
    internal struct Context {
        let state: GenerationState
        private let yieldChunk: @Sendable (LLMStreamChunk) -> Void

        init(
            continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation,
            state: GenerationState
        ) {
            self.state = state
            self.yieldChunk = { continuation.yield($0) }
        }

        init(
            sink: MLXContinuousBatchStreamSink,
            state: GenerationState
        ) {
            self.state = state
            self.yieldChunk = { sink.yield($0) }
        }

        func yield(_ chunk: LLMStreamChunk) {
            yieldChunk(chunk)
        }
    }

    internal static func append(
        _ text: String,
        context: Context
    ) -> GenerateDisposition {
        guard !text.isEmpty else {
            return .more
        }
        guard var stopDetector = context.state.stopDetector else {
            yield(text, context: context)
            return .more
        }

        let result = stopDetector.append(text)
        context.state.stopDetector = stopDetector

        switch result {
        case .more(let safeText):
            yield(safeText, context: context)
            return .more

        case .stop(let safeText):
            yield(safeText, context: context)
            context.state.stopReason = .stopSequence
            return .stop
        }
    }

    internal static func flush(context: Context) {
        guard var stopDetector = context.state.stopDetector else {
            return
        }
        let pendingText = stopDetector.flush()
        context.state.stopDetector = stopDetector
        yield(pendingText, context: context)
    }

    internal static func yield(
        _ text: String,
        context: Context
    ) {
        guard !text.isEmpty else {
            return
        }
        let firstTokenIndex = context.state.streamedTokenCount
        let tokenCount = max(
            context.state.generatedTokenCount - context.state.streamedTokenCount,
            1
        )
        context.state.streamedTokenCount += tokenCount
        context.state.generatedText += text
        context.yield(LLMStreamChunk(
            text: text,
            event: .text,
            tokenCount: tokenCount,
            tokenIDs: Array(context.state.allTokens.dropFirst(firstTokenIndex).prefix(tokenCount))
        ))
    }
}
