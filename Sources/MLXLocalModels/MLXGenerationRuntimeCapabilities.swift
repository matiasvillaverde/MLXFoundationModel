public struct MLXGenerationRuntimeCapabilities: Equatable, Hashable, Sendable {
    public let supportsContinuousBatching: Bool

    public static let scalar = Self(supportsContinuousBatching: false)
    public static let continuousBatching = Self(supportsContinuousBatching: true)

    public init(supportsContinuousBatching: Bool) {
        self.supportsContinuousBatching = supportsContinuousBatching
    }
}
