internal enum MLXContinuousBatchDecodeStepError: Error, Equatable, Sendable {
    case cacheRowCountMismatch(expected: Int, actual: Int)
    case emptyRows
    case invalidTokenColumnCount(Int)
    case invalidTokenRank(Int)
    case rowCountMismatch(expected: Int, actual: Int)
}
