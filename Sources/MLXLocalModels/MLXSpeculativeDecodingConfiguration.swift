internal struct MLXSpeculativeDecodingConfiguration: Sendable {
    let draftContext: ModelContext
    let numDraftTokens: Int

    internal init(draftContext: ModelContext, numDraftTokens: Int = 2) {
        self.draftContext = draftContext
        self.numDraftTokens = numDraftTokens
    }
}
