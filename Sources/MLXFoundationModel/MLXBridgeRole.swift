import Foundation

/// Role used by the SDK-independent prompt renderer.
public enum MLXBridgeRole: Codable, Hashable, Sendable {
    case assistant
    case system
    case tool
    case user

    public init(from decoder: Decoder) throws {
        let value = try String(from: decoder)
        switch value {
        case "assistant":
            self = .assistant

        case "system":
            self = .system

        case "tool":
            self = .tool

        case "user":
            self = .user

        default:
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported MLX bridge role.")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        try codingValue.encode(to: encoder)
    }

    var codingValue: String {
        switch self {
        case .assistant:
            "assistant"

        case .system:
            "system"

        case .tool:
            "tool"

        case .user:
            "user"
        }
    }
}
