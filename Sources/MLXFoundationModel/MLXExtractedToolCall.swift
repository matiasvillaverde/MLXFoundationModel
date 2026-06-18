import Foundation

/// A best-effort tool call parsed from model text.
public struct MLXExtractedToolCall: Equatable, Hashable, Sendable {
    public let name: String
    public let argumentsJSON: String

    public init(name: String, argumentsJSON: String) {
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}
