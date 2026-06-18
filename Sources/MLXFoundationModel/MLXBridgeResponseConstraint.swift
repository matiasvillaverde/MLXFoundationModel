import Foundation

/// Structured-output constraint rendered into prompts for local MLX models.
public struct MLXBridgeResponseConstraint: Codable, Equatable, Hashable, Sendable {
    public let jsonSchema: String
    public let instructions: String?

    public init(jsonSchema: String, instructions: String? = nil) {
        self.jsonSchema = jsonSchema
        self.instructions = instructions
    }
}
