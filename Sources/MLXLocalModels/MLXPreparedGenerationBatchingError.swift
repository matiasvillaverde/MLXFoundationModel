internal enum MLXPreparedGenerationBatchingError: Error, Equatable, Sendable {
    case emptyPrompt
    case promptCacheReuseUnsupported(reusedTokenCount: Int)
    case speculativeDecodingUnsupported
}
