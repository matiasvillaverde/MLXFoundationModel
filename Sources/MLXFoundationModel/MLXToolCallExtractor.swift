import Foundation

/// Extracts model-emitted tool calls from local model output.
public enum MLXToolCallExtractor {
    typealias Parser = MLXToolCallParsingSupport
    typealias JSONObject = Parser.JSONObject

    private struct ExtractionStrategy: Sendable {
        let name: String
        let parse: @Sendable (String) -> [MLXExtractedToolCall]
    }

    /// Extract a single tool call from generated text.
    public static func extract(from text: String) -> MLXExtractedToolCall? {
        extractAll(from: text).first
    }

    /// Extract a single tool call from generated text, normalizing arguments against tool schemas.
    public static func extract(
        from text: String,
        tools: [MLXBridgeToolDefinition]
    ) -> MLXExtractedToolCall? {
        extractAll(from: text, tools: tools).first
    }

    /// Extract all tool calls from generated text.
    public static func extractAll(from text: String) -> [MLXExtractedToolCall] {
        extractAllUnnormalized(from: text)
    }

    /// Extract all tool calls from generated text, normalizing arguments against tool schemas.
    public static func extractAll(
        from text: String,
        tools: [MLXBridgeToolDefinition]
    ) -> [MLXExtractedToolCall] {
        guard !tools.isEmpty else {
            return extractAllUnnormalized(from: text)
        }
        return extractAllThinkingAware(from: text, tools: tools)
    }

    static func extractAllUnnormalized(from text: String) -> [MLXExtractedToolCall] {
        if usesGemmaToolMarkers(text) {
            return extractGemmaToolCalls(from: text)
        }

        for strategy in strategies {
            let calls = strategy.parse(text)
            if !calls.isEmpty {
                return calls
            }
        }
        return []
    }

    private static func usesGemmaToolMarkers(_ text: String) -> Bool {
        text.contains("<|tool_call>") || text.contains("<start_function_call>")
    }

    private static let strategies: [ExtractionStrategy] = [
        .init(name: "harmony", parse: extractHarmonyToolCalls),
        .init(name: "deepSeekDSML", parse: extractDeepSeekDSMLToolCalls),
        .init(name: "xml", parse: extractXMLToolCalls),
        .init(name: "bareGLM", parse: MLXBareGLMToolCallScanner.extractCalls),
        .init(name: "minimaxM3", parse: extractMiniMaxM3ToolCalls),
        .init(name: "namespacedXML", parse: extractNamespacedToolCalls),
        .init(name: "kimiK2", parse: extractKimiToolCalls),
        .init(name: "longcat", parse: extractLongCatToolCalls),
        .init(name: "cohereAction", parse: extractCohereActionToolCalls),
        .init(name: "hermes", parse: extractHermesToolCalls),
        .init(name: "mistral", parse: extractMistralToolCalls),
        .init(name: "bracket", parse: extractBracketToolCalls),
        .init(name: "gemma", parse: extractGemmaToolCalls),
        .init(name: "json", parse: extractJSONCalls)
    ]

    static func extractJSONCalls(from text: String) -> [MLXExtractedToolCall] {
        var calls: [MLXExtractedToolCall] = []
        if let jsonText = MLXJSONTextExtractor.firstJSONObject(in: text) {
            calls.append(contentsOf: extractJSONToolCalls(from: jsonText))
        }

        guard let arrayText = Parser.firstJSONValueText(in: text, opener: "[", closer: "]"),
            let value = Parser.parseJSON(arrayText) as? [Any] else {
            return calls
        }
        calls.append(contentsOf: extractToolCalls(from: value))
        return calls
    }

    static func extractJSONToolCalls(from text: String) -> [MLXExtractedToolCall] {
        guard let object = Parser.parseJSON(text) as? JSONObject else {
            return []
        }
        return extractToolCalls(from: object)
    }

    static func extractToolCalls(from value: Any) -> [MLXExtractedToolCall] {
        if let object = value as? JSONObject {
            return extractToolCalls(from: object)
        }
        if let values = value as? [Any] {
            return values.flatMap(extractToolCalls)
        }
        return []
    }

    static func extractToolCalls(from object: JSONObject) -> [MLXExtractedToolCall] {
        let openAICalls = extractOpenAIStyleCalls(from: object)
        if !openAICalls.isEmpty {
            return openAICalls
        }
        return extractFlatCall(from: object).map { [$0] } ?? []
    }

    private static func extractFlatCall(from object: JSONObject) -> MLXExtractedToolCall? {
        if let function = object["function"] as? JSONObject {
            return extractFlatCall(from: function)
        }
        let name = object["tool_name"] as? String ?? object["name"] as? String
        guard let name else {
            return nil
        }
        let arguments = object["arguments"] ?? object["parameters"] ?? object["input"] ?? [:]
        return MLXExtractedToolCall(
            name: name,
            argumentsJSON: Parser.canonicalArgumentsJSONString(arguments)
        )
    }

    private static func extractOpenAIStyleCalls(from object: JSONObject) -> [MLXExtractedToolCall] {
        guard
            let calls = object["tool_calls"] as? [JSONObject],
            !calls.isEmpty
        else {
            return []
        }
        return calls.compactMap { call in
            guard
                let function = call["function"] as? JSONObject,
                let name = function["name"] as? String
            else {
                return nil
            }
            let arguments = function["arguments"] ?? [:]
            return MLXExtractedToolCall(
                name: name,
                argumentsJSON: Parser.canonicalArgumentsJSONString(arguments)
            )
        }
    }
}
