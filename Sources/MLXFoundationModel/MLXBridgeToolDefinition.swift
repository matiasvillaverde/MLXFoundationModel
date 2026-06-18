import Foundation

/// Tool definition used by the SDK-independent prompt renderer.
public struct MLXBridgeToolDefinition: Codable, Equatable, Hashable, Sendable {
    public let name: String
    public let description: String
    public let parametersJSONSchema: String

    public init(name: String, description: String, parametersJSONSchema: String) {
        self.name = name
        self.description = description
        self.parametersJSONSchema = parametersJSONSchema
    }
}
