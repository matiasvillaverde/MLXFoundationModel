import Foundation

/// Capabilities advertised by a local MLX model bridge.
public struct MLXModelCapabilities: Codable, Equatable, Hashable, Sendable {
    public let toolCalling: Bool
    public let structuredOutput: Bool
    public let vision: Bool
    public let reasoning: Bool

    public init(
        toolCalling: Bool = true,
        structuredOutput: Bool = false,
        vision: Bool = false,
        reasoning: Bool = false
    ) {
        self.toolCalling = toolCalling
        self.structuredOutput = structuredOutput
        self.vision = vision
        self.reasoning = reasoning
    }

    public static let text = Self()
}
