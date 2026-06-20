import Foundation

/// A local MLX model that can be used through the bridge.
public struct MLXModel: Codable, Equatable, Hashable, Sendable {
    public let id: String
    public let location: URL
    public let promptStyle: MLXPromptStyle
    public let capabilities: MLXModelCapabilities
    public let profile: MLXModelProfile?

    public init(
        id: String,
        location: URL,
        promptStyle: MLXPromptStyle? = nil,
        capabilities: MLXModelCapabilities? = nil,
        profile: MLXModelProfile? = nil
    ) {
        self.id = id
        self.location = location
        self.promptStyle = promptStyle ?? profile?.promptStyle ?? .plain
        self.capabilities = capabilities ?? profile?.capabilities ?? .text
        self.profile = profile
    }

    public static func profiled(
        id: String,
        location: URL
    ) throws -> Self {
        let profile = try MLXModelProfile.load(from: location, id: id)
        return Self(id: id, location: location, profile: profile)
    }
}
