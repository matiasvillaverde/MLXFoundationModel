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

        if var detokenizer = tokenContext.state.detokenizer {
            detokenizer.append(token: token)
            let text = detokenizer.next()
            tokenContext.state.detokenizer = detokenizer

            if let text, processGeneratedText(text, tokenContext: tokenContext) == .stop {
                return .stop
            }
        } else {
            let text = tokenContext.context.tokenizer.decode(tokens: [token])
            yieldText(text, tokenContext: tokenContext)
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
        guard !text.isEmpty else {
            return .more
        }
        guard var stopDetector = tokenContext.state.stopDetector else {
            yieldText(text, tokenContext: tokenContext)
            return .more
        }

        let result = stopDetector.append(text)
        tokenContext.state.stopDetector = stopDetector

        switch result {
        case .more(let safeText):
            yieldText(safeText, tokenContext: tokenContext)
            return .more

        case .stop(let safeText):
            yieldText(safeText, tokenContext: tokenContext)
            tokenContext.state.stopReason = .stopSequence
            return .stop
        }
    }

    nonisolated func flushPendingText(tokenContext: TokenContext) {
        guard var stopDetector = tokenContext.state.stopDetector else {
            return
        }
        let pendingText = stopDetector.flush()
        tokenContext.state.stopDetector = stopDetector
        yieldText(pendingText, tokenContext: tokenContext)
    }

    nonisolated func yieldText(_ text: String, tokenContext: TokenContext) {
        guard !text.isEmpty else {
            return
        }
        tokenContext.state.generatedText += text
        tokenContext.continuation.yield(LLMStreamChunk(text: text, event: .text))
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
