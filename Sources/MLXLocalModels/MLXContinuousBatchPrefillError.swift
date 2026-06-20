internal enum MLXContinuousBatchPrefillError: Error, Equatable {
    case emptyPrompt(rowIndex: Int)
    case emptyRequests
    case incompatibleCacheParameters(rowIndex: Int)
    case incompatiblePrefixCache(rowIndex: Int)
    case invalidPrefixCacheState(rowIndex: Int, layerIndex: Int)
    case mismatchedPrefixCacheLayerCount(rowIndex: Int)
    case mismatchedPrefixCacheOffset(rowIndex: Int, layerIndex: Int)
    case mismatchedPrefixCacheShape(rowIndex: Int, layerIndex: Int)
    case mismatchedPromptTokenCount(expected: Int, actual: Int, rowIndex: Int)
    case unsupportedPrefixCacheType(typeName: String)
}
