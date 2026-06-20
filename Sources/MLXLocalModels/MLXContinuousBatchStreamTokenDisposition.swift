internal enum MLXContinuousBatchStreamTokenDisposition: Equatable, Sendable {
    case finish(MLXContinuousBatchFinishReason)
    case streamed
    case suppressed
}
