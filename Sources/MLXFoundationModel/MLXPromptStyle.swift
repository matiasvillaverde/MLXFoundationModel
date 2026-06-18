import Foundation

/// Prompt format used when converting a Foundation Models transcript into text
/// for an instruction-tuned MLX model.
public enum MLXPromptStyle: Codable, CaseIterable, Hashable, Sendable {
    /// ChatML-style role markers used by Qwen and related instruction models.
    case chatML

    /// Plain role-prefixed text.
    case plain

    public init(from decoder: Decoder) throws {
        let value = try String(from: decoder)
        switch value {
        case "chatML":
            self = .chatML

        case "plain":
            self = .plain

        default:
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported MLX prompt style.")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        try codingValue.encode(to: encoder)
    }

    var codingValue: String {
        switch self {
        case .chatML:
            "chatML"

        case .plain:
            "plain"
        }
    }
}
