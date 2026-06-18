import Foundation

/// Message used by the SDK-independent prompt renderer.
public struct MLXBridgeMessage: Codable, Equatable, Hashable, Sendable {
    public let role: MLXBridgeRole
    public let content: String
    public let name: String?

    public init(role: MLXBridgeRole, content: String, name: String? = nil) {
        self.role = role
        self.content = content
        self.name = name
    }
}
