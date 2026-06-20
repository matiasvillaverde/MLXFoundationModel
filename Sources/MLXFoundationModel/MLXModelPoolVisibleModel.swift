/// API-visible model descriptor emitted by ``MLXModelPool`` snapshots.
public struct MLXModelPoolVisibleModel: Equatable, Hashable, Sendable {
    /// Public identifier accepted by the pool.
    public let id: String

    /// Canonical base model identifier when this descriptor is a serving profile.
    public let sourceModelID: String?

    /// Public aliases that resolve to ``id``.
    public let aliases: [String]

    /// Whether this descriptor is backed by a serving profile variant.
    public let isServingProfile: Bool

    /// Prompt style used by the resolved model.
    public let promptStyle: MLXPromptStyle

    /// Capabilities advertised by the resolved model.
    public let capabilities: MLXModelCapabilities

    /// Runtime family needed for the resolved model.
    public let runtimeKind: MLXModelRuntimeKind

    /// Declared model context length, when known.
    public let contextLength: Int?

    /// Default response-token limit after profile overrides are applied.
    public let maximumResponseTokens: Int?

    public init(
        id: String,
        sourceModelID: String?,
        aliases: [String],
        isServingProfile: Bool,
        promptStyle: MLXPromptStyle,
        capabilities: MLXModelCapabilities,
        runtimeKind: MLXModelRuntimeKind,
        contextLength: Int?,
        maximumResponseTokens: Int?
    ) {
        self.id = id
        self.sourceModelID = sourceModelID
        self.aliases = aliases
        self.isServingProfile = isServingProfile
        self.promptStyle = promptStyle
        self.capabilities = capabilities
        self.runtimeKind = runtimeKind
        self.contextLength = contextLength
        self.maximumResponseTokens = maximumResponseTokens
    }
}
