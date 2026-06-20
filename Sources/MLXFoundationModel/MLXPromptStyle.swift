import Foundation

/// Prompt format used when converting a Foundation Models transcript into text
/// for an instruction-tuned MLX model.
public enum MLXPromptStyle: Codable, CaseIterable, Hashable, Sendable {
    /// ChatML-style role markers used by Qwen and related instruction models.
    case chatML

    /// Cohere action-marker tool format.
    case cohereAction

    /// DeepSeek DSML tool-call format.
    case deepSeekDSML

    /// Legacy Gemma function-call markers.
    case functionGemma

    /// Gemma 4 tool-call markers.
    case gemma

    /// GLM XML tool-call format.
    case glmXML

    /// Harmony channel format used by gpt-oss models.
    case harmony

    /// Kimi K2 tool-call section format.
    case kimiK2

    /// LongCat XML tool-call format.
    case longCat

    /// MiniMax-M3 native prompt and invoke format.
    case minimaxM3

    /// MiniMax XML invoke format.
    case minimaxXML

    /// Mistral tool-call marker format.
    case mistralToolCall

    /// Plain role-prefixed text.
    case plain

    /// Qwen XML function-call format.
    case qwenXML

    private static let decodedStyles: [String: Self] = [
        "chatML": .chatML,
        "cohereAction": .cohereAction,
        "deepSeekDSML": .deepSeekDSML,
        "functionGemma": .functionGemma,
        "gemma": .gemma,
        "glmXML": .glmXML,
        "harmony": .harmony,
        "kimiK2": .kimiK2,
        "longCat": .longCat,
        "minimaxM3": .minimaxM3,
        "minimaxXML": .minimaxXML,
        "mistralToolCall": .mistralToolCall,
        "plain": .plain,
        "qwenXML": .qwenXML
    ]

    private static let encodedStyles = Dictionary(uniqueKeysWithValues: decodedStyles.map { key, value in
        (value, key)
    })

    public init(from decoder: Decoder) throws {
        let value = try String(from: decoder)
        guard let style = Self.decodedStyles[value] else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported MLX prompt style.")
            )
        }
        self = style
    }

    public func encode(to encoder: Encoder) throws {
        try codingValue.encode(to: encoder)
    }

    public var codingValue: String {
        Self.encodedStyles[self] ?? "plain"
    }
}
