internal enum MLXContinuousBatchLogitRowsError: Error, Equatable, Sendable {
    case invalidLogitRank(Int)
    case rowCountMismatch(expected: Int, actual: Int)
}
