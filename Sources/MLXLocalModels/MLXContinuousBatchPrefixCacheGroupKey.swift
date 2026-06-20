import Foundation

internal enum MLXContinuousBatchPrefixCacheGroupKey: Equatable, Sendable {
    case mergeable(cachedTokenCount: Int, layout: [String])
    case singleton(UUID)
    case uncached
}
