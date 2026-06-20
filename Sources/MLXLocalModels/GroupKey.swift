internal struct GroupKey: Equatable {
    let cacheSignature: MLXContinuousBatchCacheSignature
    let promptTokenCount: Int
    let prefixCacheGroupKey: MLXContinuousBatchPrefixCacheGroupKey

    init(request: MLXContinuousBatchPrefillRequest) {
        self.cacheSignature = MLXContinuousBatchCacheSignature(parameters: request.parameters)
        self.promptTokenCount = request.promptTokenIDs.count
        self.prefixCacheGroupKey = request.prefixCacheGroupKey
    }
}
