import Foundation

/// Intermediate request shape shared by tests and the Foundation Models adapter.
public struct MLXBridgeRequest: Codable, Equatable, Hashable, Sendable {
    public let instructions: String?
    public let messages: [MLXBridgeMessage]
    public let responseConstraint: MLXBridgeResponseConstraint?
    public let tools: [MLXBridgeToolDefinition]

    public init(
        messages: [MLXBridgeMessage],
        instructions: String? = nil,
        responseConstraint: MLXBridgeResponseConstraint? = nil,
        tools: [MLXBridgeToolDefinition] = []
    ) {
        self.instructions = instructions
        self.messages = messages
        self.responseConstraint = responseConstraint
        self.tools = tools
    }
}
