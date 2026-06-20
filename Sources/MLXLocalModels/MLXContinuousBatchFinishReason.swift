internal enum MLXContinuousBatchFinishReason: Equatable, Sendable {
    case maximumTokenCount
    case stopToken(Int)
    case streamRequestedStop(Int)
}
