import Foundation

/// Extracts simple JSON tool calls from local model output.
public enum MLXToolCallExtractor {
    /// Extract a single JSON tool call from generated text.
    public static func extract(from text: String) -> MLXExtractedToolCall? {
        guard let jsonText = MLXJSONTextExtractor.firstJSONObject(in: text) else {
            return nil
        }
        guard
            let data = jsonText.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        if let call = extractOpenAIStyleCall(from: object) {
            return call
        }
        return extractFlatCall(from: object)
    }

    private static func extractFlatCall(from object: [String: Any]) -> MLXExtractedToolCall? {
        let name = object["tool_name"] as? String ?? object["name"] as? String
        guard let name else {
            return nil
        }
        let arguments = object["arguments"] ?? object["input"] ?? [:]
        return MLXExtractedToolCall(
            name: name,
            argumentsJSON: canonicalJSONString(arguments)
        )
    }

    private static func extractOpenAIStyleCall(from object: [String: Any]) -> MLXExtractedToolCall? {
        guard
            let calls = object["tool_calls"] as? [[String: Any]],
            let first = calls.first,
            let function = first["function"] as? [String: Any],
            let name = function["name"] as? String
        else {
            return nil
        }
        let arguments = function["arguments"] ?? [:]
        if let argumentsString = arguments as? String {
            return MLXExtractedToolCall(name: name, argumentsJSON: argumentsString)
        }
        return MLXExtractedToolCall(name: name, argumentsJSON: canonicalJSONString(arguments))
    }

    private static func canonicalJSONString(_ value: Any) -> String {
        guard
            JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let string = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return string
    }
}
