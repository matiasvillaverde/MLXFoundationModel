import Foundation

/// A local MLX model that can be used through the bridge.
public struct MLXModel: Codable, Equatable, Hashable, Sendable {
    public let id: String
    public let location: URL
    public let promptStyle: MLXPromptStyle
    public let capabilities: MLXModelCapabilities

    public init(
        id: String,
        location: URL,
        promptStyle: MLXPromptStyle = .plain,
        capabilities: MLXModelCapabilities = .text
    ) {
        self.id = id
        self.location = location
        self.promptStyle = promptStyle
        self.capabilities = capabilities
    }
}
