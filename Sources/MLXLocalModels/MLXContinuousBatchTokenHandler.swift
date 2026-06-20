import Foundation
import Tokenizers

internal struct MLXContinuousBatchTokenHandler: Sendable {
    private let decodeToken: @Sendable (Int) -> String
    private let emitterContext: MLXStreamTextEmitter.Context
    private let now: @Sendable () -> ContinuousClock.Instant
    private let state: GenerationState

    internal init(
        state: GenerationState,
        sink: MLXContinuousBatchStreamSink,
        tokenizer: any Tokenizer,
        now: @escaping @Sendable () -> ContinuousClock.Instant = { ContinuousClock().now }
    ) {
        self.decodeToken = { tokenizer.decode(tokens: [$0]) }
        self.emitterContext = .init(sink: sink, state: state)
        self.now = now
        self.state = state
        if state.detokenizer == nil {
            state.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        }
    }

    internal init(
        state: GenerationState,
        sink: MLXContinuousBatchStreamSink,
        decodeToken: @escaping @Sendable (Int) -> String,
        now: @escaping @Sendable () -> ContinuousClock.Instant = { ContinuousClock().now }
    ) {
        self.decodeToken = decodeToken
        self.emitterContext = .init(sink: sink, state: state)
        self.now = now
        self.state = state
    }

    internal func flush() {
        MLXStreamTextEmitter.flush(context: emitterContext)
    }

    internal func stream(
        tokenID: Int
    ) -> MLXContinuousBatchStreamTokenDisposition {
        prepareState(tokenID: tokenID)
        let streamedTokenCount = state.streamedTokenCount
        let disposition = streamDecodedText(for: tokenID)
        if disposition == .stop {
            return .finish(.streamRequestedStop(tokenID))
        }
        return state.streamedTokenCount > streamedTokenCount ? .streamed : .suppressed
    }

    private func prepareState(tokenID: Int) {
        if state.firstTokenTime == nil {
            state.firstTokenTime = now()
        }
        state.allTokens.append(tokenID)
        state.generatedTokenCount += 1
        MLXGenerationDiagnostics.recordGeneratedToken(
            tokenID: tokenID,
            tokenText: decodeToken(tokenID),
            index: state.generatedTokenCount
        )
    }

    private func streamDecodedText(for tokenID: Int) -> GenerateDisposition {
        if var detokenizer = state.detokenizer {
            detokenizer.append(token: tokenID)
            let text = detokenizer.next()
            state.detokenizer = detokenizer
            guard let text else {
                return .more
            }
            return MLXStreamTextEmitter.append(text, context: emitterContext)
        }
        return MLXStreamTextEmitter.append(decodeToken(tokenID), context: emitterContext)
    }
}
