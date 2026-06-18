import Foundation

/// Finds JSON object payloads inside local model output.
public enum MLXJSONTextExtractor {
    /// Return the first valid JSON object contained in generated text.
    public static func firstJSONObject(in text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let unfenced = stripOuterCodeFence(trimmed)
        return scanForJSONObject(in: unfenced) ?? scanForJSONObject(in: trimmed)
    }

    private static func scanForJSONObject(in text: String) -> String? {
        var searchStart = text.startIndex
        while let openingBrace = text[searchStart...].firstIndex(of: "{") {
            if let object = firstJSONObject(startingAt: openingBrace, in: text) {
                return object
            }
            searchStart = text.index(after: openingBrace)
        }
        return nil
    }

    private static func firstJSONObject(
        startingAt start: String.Index,
        in text: String
    ) -> String? {
        var end = text.index(after: start)
        while end <= text.endIndex {
            let candidate = String(text[start..<end])
            if isJSONObject(candidate) {
                return candidate
            }
            guard end < text.endIndex else {
                return nil
            }
            end = text.index(after: end)
        }
        return nil
    }

    private static func isJSONObject(_ text: String) -> Bool {
        guard
            let data = text.data(using: .utf8),
            (try? JSONSerialization.jsonObject(with: data)) is [String: Any]
        else {
            return false
        }
        return true
    }

    private static func stripOuterCodeFence(_ text: String) -> String {
        guard text.hasPrefix("```") else {
            return text
        }
        var lines = text.components(separatedBy: .newlines)
        guard lines.count >= 3 else {
            return text
        }
        lines.removeFirst()
        if lines.last?.trimmingCharacters(in: .whitespacesAndNewlines) == "```" {
            lines.removeLast()
        }
        return lines.joined(separator: "\n")
    }
}
