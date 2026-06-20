extension MLXSession {
    nonisolated func processToken(
        token: Int,
        tokenContext: TokenContext
    ) -> GenerateDisposition {
        if tokenContext.state.firstTokenTime == nil {
            tokenContext.state.firstTokenTime = tokenContext.clock.now
        }

        tokenContext.state.allTokens.append(token)
        tokenContext.state.generatedTokenCount += 1
        let tokenText = tokenContext.context.tokenizer.decode(tokens: [token])
        MLXGenerationDiagnostics.recordGeneratedToken(
            tokenID: token,
            tokenText: tokenText,
            index: tokenContext.state.generatedTokenCount
        )

        if var detokenizer = tokenContext.state.detokenizer {
            detokenizer.append(token: token)
            let text = detokenizer.next()
            tokenContext.state.detokenizer = detokenizer

            if let text, processGeneratedText(text, tokenContext: tokenContext) == .stop {
                return .stop
            }
        } else {
            yieldText(tokenText, tokenContext: tokenContext)
        }

        if tokenContext.state.generatedTokenCount >= tokenContext.input.limits.maxTokens {
            tokenContext.state.stopReason = .maxTokens
            return .stop
        }

        return .more
    }

    nonisolated func processGeneratedText(
        _ text: String,
        tokenContext: TokenContext
    ) -> GenerateDisposition {
        MLXStreamTextEmitter.append(
            text,
            context: textEmissionContext(tokenContext)
        )
    }

    nonisolated func flushPendingText(tokenContext: TokenContext) {
        MLXStreamTextEmitter.flush(context: textEmissionContext(tokenContext))
    }

    nonisolated func yieldText(_ text: String, tokenContext: TokenContext) {
        MLXStreamTextEmitter.yield(
            text,
            context: textEmissionContext(tokenContext)
        )
    }

    nonisolated private func textEmissionContext(
        _ tokenContext: TokenContext
    ) -> MLXStreamTextEmitter.Context {
        MLXStreamTextEmitter.Context(
            continuation: tokenContext.continuation,
            state: tokenContext.state
        )
    }

    nonisolated func isStopToken(_ token: Int, context: ModelContext) -> Bool {
        if token == context.tokenizer.unknownTokenId || token == context.tokenizer.eosTokenId {
            return true
        }
        if context.configuration.eosTokenIds.contains(token) {
            return true
        }

        let additionalEOSTokenIds = Set(
            context.configuration.extraEOSTokens.compactMap { eosToken in
                context.tokenizer.convertTokenToId(eosToken)
            }
        )
        return additionalEOSTokenIds.contains(token)
    }
}
