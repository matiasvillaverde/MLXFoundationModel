import Foundation

extension MLXToolCallExtractor {
    private static let thinkingMarkerPairs = [
        ("<think>", "</think>"),
        ("<mm:think>", "</mm:think>"),
        ("<longcat_think>", "</longcat_think>"),
        ("<|channel>thought\n", "<channel|>"),
        ("<|channel|>analysis<|message|>", "<|end|>")
    ]

    static func extractAllThinkingAware(
        from text: String,
        tools: [MLXBridgeToolDefinition]
    ) -> [MLXExtractedToolCall] {
        let split = splitThinkingContent(in: text)
        let regularCalls = extractAllUnnormalized(from: split.regular)
        if !regularCalls.isEmpty {
            return MLXToolArgumentNormalizer.normalize(regularCalls, using: tools)
        }

        let thinkingCalls = extractAllUnnormalized(from: split.thinking)
        return validToolCalls(thinkingCalls, tools: tools)
    }

    private static func splitThinkingContent(in text: String) -> (regular: String, thinking: String) {
        thinkingMarkerPairs.reduce((regular: text, thinking: "")) { partial, markers in
            let result = extractingThinkingBlocks(
                from: partial.regular,
                start: markers.0,
                end: markers.1
            )
            return (
                regular: result.regular,
                thinking: joinedThinking([partial.thinking, result.thinking])
            )
        }
    }

    private static func extractingThinkingBlocks(
        from text: String,
        start: String,
        end: String
    ) -> (regular: String, thinking: String) {
        var regular = ""
        var thinking: [String] = []
        var cursor = text.startIndex

        while let startRange = text.range(of: start, range: cursor..<text.endIndex),
            let endRange = text.range(of: end, range: startRange.upperBound..<text.endIndex) {
            regular += text[cursor..<startRange.lowerBound]
            thinking.append(String(text[startRange.upperBound..<endRange.lowerBound]))
            cursor = endRange.upperBound
        }

        regular += text[cursor..<text.endIndex]
        return (regular, joinedThinking(thinking))
    }

    private static func validToolCalls(
        _ calls: [MLXExtractedToolCall],
        tools: [MLXBridgeToolDefinition]
    ) -> [MLXExtractedToolCall] {
        let normalized = MLXToolArgumentNormalizer.normalize(calls, using: tools)
        let validNames = Set(tools.map(\.name))
        return normalized.filter { validNames.contains($0.name) }
    }

    private static func joinedThinking(_ values: [String]) -> String {
        values
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }
}
