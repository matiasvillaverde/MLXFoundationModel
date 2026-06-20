extension MLXSession {
    internal func makeTokenSamplingConfiguration(
        from container: ModelContainer,
        drySequenceBreakers: [String] = [],
        reasoningEndMarker: String? = nil
    ) async -> (
        suppressTokenIds: Set<Int>,
        xtcProtectedTokenIds: Set<Int>,
        drySequenceBreakerTokenIds: [[Int]],
        reasoningEndTokenIds: [Int]
    ) {
        await container.perform { context in
            var protectedTokenIds = context.configuration.eosTokenIds
            if let eosTokenId = context.tokenizer.eosTokenId {
                protectedTokenIds.insert(eosTokenId)
            }
            if let unknownTokenId = context.tokenizer.unknownTokenId {
                protectedTokenIds.insert(unknownTokenId)
            }
            protectedTokenIds.formUnion(
                context.configuration.extraEOSTokens.compactMap { token in
                    context.tokenizer.convertTokenToId(token)
                }
            )

            let dryBreakerTokenIds = drySequenceBreakers.compactMap { breaker in
                let tokenIds = context.tokenizer.encode(
                    text: breaker,
                    addSpecialTokens: false
                )
                return tokenIds.isEmpty ? nil : tokenIds
            }

            let reasoningEndTokenIds: [Int]
            if let reasoningEndMarker, !reasoningEndMarker.isEmpty {
                reasoningEndTokenIds = context.tokenizer.encode(
                    text: reasoningEndMarker,
                    addSpecialTokens: false
                )
            } else {
                reasoningEndTokenIds = []
            }

            return (
                context.configuration.suppressTokenIds,
                protectedTokenIds,
                dryBreakerTokenIds,
                reasoningEndTokenIds
            )
        }
    }
}
