internal struct MLXContinuousBatchGenerationRow: Equatable, Sendable {
    internal private(set) var generatedTokenCount: Int
    internal private(set) var previousTokenID: Int
    internal let maximumTokenCount: Int
    internal let stopTokenIDs: Set<Int>

    internal init(
        previousTokenID: Int,
        maximumTokenCount: Int,
        generatedTokenCount: Int = 0,
        stopTokenIDs: Set<Int> = []
    ) {
        self.previousTokenID = previousTokenID
        self.maximumTokenCount = max(1, maximumTokenCount)
        self.generatedTokenCount = max(0, generatedTokenCount)
        self.stopTokenIDs = stopTokenIDs
    }

    internal mutating func accept(tokenID: Int) -> MLXContinuousBatchFinishReason? {
        previousTokenID = tokenID
        generatedTokenCount += 1

        if stopTokenIDs.contains(tokenID) {
            return .stopToken(tokenID)
        }
        if generatedTokenCount >= maximumTokenCount {
            return .maximumTokenCount
        }
        return nil
    }
}
