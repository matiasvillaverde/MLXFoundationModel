import MLX
import Tokenizers

internal struct MLXContinuousBatchPrefillRequest: @unchecked Sendable {
    internal let promptTokenIDs: [Int]
    internal let processorPromptTokenIDs: [Int]
    internal let prefixCache: MLXContinuousBatchPrefixCache?
    internal let promptCacheStorage: MLXContinuousBatchPromptCacheStorage?
    internal let parameters: GenerateParameters
    internal let stopTokenIDs: Set<Int>
    internal let sink: MLXContinuousBatchStreamSink
    internal var prefixCacheGroupKey: MLXContinuousBatchPrefixCacheGroupKey {
        prefixCache?.groupKey ?? .uncached
    }
    private let handleToken: @Sendable (
        Int,
        MLXContinuousBatchStreamSink
    ) -> MLXContinuousBatchStreamTokenDisposition

    internal init(
        promptTokenIDs: [Int],
        parameters: GenerateParameters,
        tokenText: @escaping @Sendable (Int) -> String,
        sink: MLXContinuousBatchStreamSink,
        stopTokenIDs: Set<Int> = [],
        processorPromptTokenIDs: [Int] = [],
        prefixCache: MLXContinuousBatchPrefixCache? = nil,
        promptCacheStorage: MLXContinuousBatchPromptCacheStorage? = nil
    ) {
        self.promptTokenIDs = promptTokenIDs
        self.processorPromptTokenIDs = Self.processorTokens(
            processorPromptTokenIDs,
            fallback: promptTokenIDs
        )
        self.prefixCache = prefixCache
        self.promptCacheStorage = promptCacheStorage
        self.parameters = parameters
        self.stopTokenIDs = stopTokenIDs
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
        promptTokenIDs: [Int],
        parameters: GenerateParameters,
        state: GenerationState,
        tokenizer: any Tokenizer,
        sink: MLXContinuousBatchStreamSink,
        stopTokenIDs: Set<Int> = [],
        processorPromptTokenIDs: [Int] = [],
        prefixCache: MLXContinuousBatchPrefixCache? = nil,
        promptCacheStorage: MLXContinuousBatchPromptCacheStorage? = nil
    ) {
        let handler = MLXContinuousBatchTokenHandler(
            state: state,
            sink: sink,
            tokenizer: tokenizer
        )
        self.init(
            promptTokenIDs: promptTokenIDs,
            parameters: parameters,
            sink: Self.flushingSink(sink, handler: handler),
            handleToken: { tokenID, _ in handler.stream(tokenID: tokenID) },
            stopTokenIDs: stopTokenIDs,
            processorPromptTokenIDs: processorPromptTokenIDs,
            prefixCache: prefixCache,
            promptCacheStorage: promptCacheStorage
        )
    }

    internal init(
        promptTokenIDs: [Int],
        parameters: GenerateParameters,
        state: GenerationState,
        decodeToken: @escaping @Sendable (Int) -> String,
        sink: MLXContinuousBatchStreamSink,
        stopTokenIDs: Set<Int> = [],
        processorPromptTokenIDs: [Int] = [],
        prefixCache: MLXContinuousBatchPrefixCache? = nil,
        promptCacheStorage: MLXContinuousBatchPromptCacheStorage? = nil
    ) {
        let handler = MLXContinuousBatchTokenHandler(
            state: state,
            sink: sink,
            decodeToken: decodeToken
        )
        self.init(
            promptTokenIDs: promptTokenIDs,
            parameters: parameters,
            sink: Self.flushingSink(sink, handler: handler),
            handleToken: { tokenID, _ in handler.stream(tokenID: tokenID) },
            stopTokenIDs: stopTokenIDs,
            processorPromptTokenIDs: processorPromptTokenIDs,
            prefixCache: prefixCache,
            promptCacheStorage: promptCacheStorage
        )
    }

    internal init(
        promptTokenIDs: [Int],
        parameters: GenerateParameters,
        sink: MLXContinuousBatchStreamSink,
        handleToken: @escaping @Sendable (
            Int,
            MLXContinuousBatchStreamSink
        ) -> MLXContinuousBatchStreamTokenDisposition,
        stopTokenIDs: Set<Int> = [],
        processorPromptTokenIDs: [Int] = [],
        prefixCache: MLXContinuousBatchPrefixCache? = nil,
        promptCacheStorage: MLXContinuousBatchPromptCacheStorage? = nil
    ) {
        self.promptTokenIDs = promptTokenIDs
        self.processorPromptTokenIDs = Self.processorTokens(
            processorPromptTokenIDs,
            fallback: promptTokenIDs
        )
        self.prefixCache = prefixCache
        self.promptCacheStorage = promptCacheStorage
        self.parameters = parameters
        self.stopTokenIDs = stopTokenIDs
        self.sink = sink
        self.handleToken = handleToken
    }

    internal func continuousBatchGenerationRequest(
        previousTokenID: Int,
        generatedTokenCount: Int
    ) -> MLXContinuousBatchGenerationRequest {
        MLXContinuousBatchGenerationRequest(
            generationRow: .init(
                previousTokenID: previousTokenID,
                maximumTokenCount: parameters.maxTokens ?? Int.max,
                generatedTokenCount: generatedTokenCount,
                stopTokenIDs: stopTokenIDs
            ),
            sink: sink,
            handleToken: handleToken
        )
    }

    internal func processor(
        grammarCompiler: GrammarConstraintCompiler?
    ) throws -> LogitProcessor? {
        var processor = try parameters.processor(grammarCompiler: grammarCompiler)
        processor?.prompt(MLXArray(processorPromptTokenIDs))
        return processor
    }

    internal func stream(
        tokenID: Int
    ) -> MLXContinuousBatchStreamTokenDisposition {
        handleToken(tokenID, sink)
    }

    private static func flushingSink(
        _ sink: MLXContinuousBatchStreamSink,
        handler: MLXContinuousBatchTokenHandler
    ) -> MLXContinuousBatchStreamSink {
        MLXContinuousBatchStreamSink(
            yield: sink.yield,
            finish: { reason in
                handler.flush()
                sink.finish(reason)
            },
            fail: sink.fail
        )
    }

    private static func processorTokens(
        _ tokens: [Int],
        fallback: [Int]
    ) -> [Int] {
        tokens.isEmpty ? fallback : tokens
    }
}
