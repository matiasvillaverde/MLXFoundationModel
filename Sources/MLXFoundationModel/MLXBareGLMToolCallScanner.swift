import Foundation

enum MLXBareGLMToolCallScanner {
    struct ParsedCall {
        let range: Range<String.Index>
        let call: MLXExtractedToolCall
    }

    enum ToolNameScope {
        case any
        case known(Set<String>)

        var sortedNames: [String] {
            switch self {
            case .any:
                []

            case .known(let names):
                names
                    .filter { !$0.isEmpty }
                    .sorted { $0.count > $1.count }
            }
        }
    }

    static let keyStart = "<arg_key>"
    static let keyEnd = "</arg_key>"
    static let valueStart = "<arg_value>"
    static let valueEnd = "</arg_value>"

    static func firstStart(
        in text: String,
        toolNames: Set<String>
    ) -> Range<String.Index>? {
        firstKnownStart(
            in: text,
            searchRange: text.startIndex..<text.endIndex,
            toolNames: toolNames
        )
    }

    static func partialStartSuffixLength(
        in text: String,
        toolNames: Set<String>
    ) -> Int? {
        let markers = toolNames
            .filter { !$0.isEmpty }
            .map { "\($0)\(keyStart)" }
        guard let maximumMarkerLength = markers.map(\.count).max(),
            maximumMarkerLength > 1 else {
            return nil
        }

        let maximum = min(text.count, maximumMarkerLength - 1)
        for length in stride(from: maximum, through: 1, by: -1) {
            let suffixStart = text.index(text.endIndex, offsetBy: -length)
            let suffix = String(text[suffixStart...])
            guard hasBoundaryBefore(suffixStart, in: text) else {
                continue
            }
            if markers.contains(where: { $0.hasPrefix(suffix) }) {
                return length
            }
        }
        return nil
    }

    static func shouldDropUnresolvedTail(
        _ text: String,
        toolNames: Set<String>
    ) -> Bool {
        guard !text.isEmpty else {
            return false
        }
        return toolNames
            .filter { !$0.isEmpty }
            .map { "\($0)\(keyStart)" }
            .contains { marker in
                marker.hasPrefix(text) || text.hasPrefix(marker)
            }
    }

    static func completedRange(
        in text: String,
        toolNames: Set<String>,
        final: Bool
    ) -> Range<String.Index>? {
        guard !toolNames.isEmpty,
            let parsed = firstParsedCall(
                in: text,
                searchRange: text.startIndex..<text.endIndex,
                scope: .known(toolNames),
                final: final
            ) else {
            return nil
        }
        return parsed.range
    }

    static func extractCalls(from text: String) -> [MLXExtractedToolCall] {
        var calls: [MLXExtractedToolCall] = []
        var cursor = text.startIndex
        while cursor < text.endIndex,
            let parsed = firstParsedCall(
                in: text,
                searchRange: cursor..<text.endIndex,
                scope: .any,
                final: true
            ) {
            calls.append(parsed.call)
            cursor = parsed.range.upperBound
        }
        return calls
    }

    static func firstKnownStart(
        in text: String,
        searchRange: Range<String.Index>,
        toolNames: Set<String>
    ) -> Range<String.Index>? {
        var best: Range<String.Index>?
        for name in ToolNameScope.known(toolNames).sortedNames {
            best = earlier(
                best,
                firstStart(in: text, searchRange: searchRange, toolName: name)
            )
        }
        return best
    }

    static func firstStart(
        in text: String,
        searchRange: Range<String.Index>,
        toolName: String
    ) -> Range<String.Index>? {
        let marker = "\(toolName)\(keyStart)"
        var cursor = searchRange.lowerBound
        while let range = text.range(of: marker, range: cursor..<searchRange.upperBound) {
            if hasBoundaryBefore(range.lowerBound, in: text) {
                return range
            }
            cursor = range.upperBound
        }
        return nil
    }

    static func hasBoundaryBefore(
        _ index: String.Index,
        in text: String
    ) -> Bool {
        guard index > text.startIndex else {
            return true
        }
        return !isToolNameBody(text[text.index(before: index)])
    }

    static func isToolNameBody(_ character: Character) -> Bool {
        character == "_"
            || character == "."
            || character == "-"
            || character == ":"
            || character.isLetter
            || character.isNumber
    }

    static func earlier(
        _ current: Range<String.Index>?,
        _ candidate: Range<String.Index>?
    ) -> Range<String.Index>? {
        guard let candidate else {
            return current
        }
        guard let current else {
            return candidate
        }
        return candidate.lowerBound < current.lowerBound ? candidate : current
    }
}
