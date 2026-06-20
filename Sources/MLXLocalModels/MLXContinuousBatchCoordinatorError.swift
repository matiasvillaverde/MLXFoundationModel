internal enum MLXContinuousBatchCoordinatorError: Error, Equatable, Sendable {
    case emptyAdmission
    case pagedKVCacheDisabled
}
