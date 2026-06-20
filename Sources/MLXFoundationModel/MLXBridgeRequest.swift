import Foundation

/// Intermediate request shape shared by tests and the Foundation Models adapter.
public struct MLXBridgeRequest: Codable, Equatable, Hashable, Sendable {
    public let instructions: String?
    public let messages: [MLXBridgeMessage]
    public let reasoningEnabled: Bool
    public let reasoningOptions: MLXBridgeReasoningOptions?
    public let responseConstraint: MLXBridgeResponseConstraint?
    public let tools: [MLXBridgeToolDefinition]

    public init(
        messages: [MLXBridgeMessage],
        instructions: String? = nil,
        reasoningEnabled: Bool = false,
        reasoningOptions: MLXBridgeReasoningOptions? = nil,
        responseConstraint: MLXBridgeResponseConstraint? = nil,
        tools: [MLXBridgeToolDefinition] = []
    ) {
        let effectiveReasoningOptions = reasoningOptions ?? MLXBridgeReasoningOptions(
            isEnabled: reasoningEnabled
        )
        self.instructions = instructions
        self.messages = messages
        self.reasoningEnabled = effectiveReasoningOptions.isEnabled
        self.reasoningOptions = reasoningOptions
        self.responseConstraint = responseConstraint
        self.tools = tools
    }

    public var effectiveReasoningOptions: MLXBridgeReasoningOptions {
        reasoningOptions ?? MLXBridgeReasoningOptions(isEnabled: reasoningEnabled)
    }
}
