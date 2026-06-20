import Foundation

enum MLXToolCallEnvelopeDetector {
    enum EnvelopeKind {
        case consumeOnly
        case mistral
        case paired(end: String)
    }

    struct Envelope {
        let range: Range<String.Index>
        let kind: EnvelopeKind
    }

    private static let fixedEnvelopes: [(start: String, end: String)] = [
        ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"),
        ("<tool_call>", "</tool_call>"),
        ("]<]minimax[>[<tool_call>", "]<]minimax[>[</tool_call>"),
        ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
        ("<|tool_call_begin|>", "<|tool_call_end|>"),
        ("<longcat_tool_call>", "</longcat_tool_call>"),
        ("<|START_ACTION|>", "<|END_ACTION|>"),
        ("<|tool_call_start|>", "<|tool_call_end|>"),
        ("<|tool_call>", "<tool_call|>"),
        ("<start_function_call>", "<end_function_call>")
    ]

    private static let mistralStart = "[TOOL_CALLS]"
    private static let consumeOnlyMarkers = [
        "<tool_call|>",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
        "<|END_ACTION|>"
    ]
    private static let allStartMarkers = fixedEnvelopes.map(\.start)
        + consumeOnlyMarkers
        + [mistralStart]

    static func firstEnvelope(in text: String) -> Envelope? {
        var result = mistralEnvelope(in: text)
        result = earlier(result, fixedEnvelope(in: text))
        result = earlier(result, consumeOnlyEnvelope(in: text))
        result = earlier(result, namespacedEnvelope(in: text))
        result = earlier(result, bracketEnvelope(in: text))
        return result
    }

    static func partialStartSuffixLength(in text: String) -> Int {
        let maximum = max(0, min(text.count, longestStartMarkerLength - 1))
        var retainedLength = 0
        guard maximum > 0 else {
            return partialNamespacedOpenSuffixLength(in: text) ?? 0
        }
        for length in stride(from: maximum, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if isPartialStartMarker(suffix) {
                retainedLength = length
                break
            }
        }
        return [
            retainedLength,
            partialNamespacedOpenSuffixLength(in: text) ?? 0,
            MLXBracketToolCallEnvelopeScanner.partialSuffixLength(in: text) ?? 0
        ].max() ?? 0
    }

    static func shouldDropUnresolvedTail(_ text: String) -> Bool {
        guard !text.isEmpty else {
            return false
        }
        if text == "]" {
            return false
        }
        if allStartMarkers.contains(where: { $0.hasPrefix(text) || text.hasPrefix($0) }) {
            return true
        }
        return isUnresolvedNamespacedToolCallTail(text)
            || MLXBracketToolCallEnvelopeScanner.shouldDropUnresolvedTail(text)
    }

    private static var longestStartMarkerLength: Int {
        allStartMarkers.map(\.count).max() ?? 0
    }

    private static func isPartialStartMarker(_ text: String) -> Bool {
        allStartMarkers.contains { $0.hasPrefix(text) }
    }

    private static func mistralEnvelope(in text: String) -> Envelope? {
        guard let range = text.range(of: mistralStart) else {
            return nil
        }
        return Envelope(range: range, kind: .mistral)
    }

    private static func fixedEnvelope(in text: String) -> Envelope? {
        fixedEnvelopes.reduce(nil) { result, fixed in
            guard let range = text.range(of: fixed.start) else {
                return result
            }
            return earlier(result, Envelope(range: range, kind: .paired(end: fixed.end)))
        }
    }

    private static func consumeOnlyEnvelope(in text: String) -> Envelope? {
        consumeOnlyMarkers.reduce(nil) { result, marker in
            guard let range = text.range(of: marker) else {
                return result
            }
            return earlier(result, Envelope(range: range, kind: .consumeOnly))
        }
    }

    private static func namespacedEnvelope(in text: String) -> Envelope? {
        var searchStart = text.startIndex
        while let openRange = text.range(of: "<", range: searchStart..<text.endIndex),
            let closeRange = text.range(of: ">", range: openRange.upperBound..<text.endIndex) {
            let tag = String(text[openRange.upperBound..<closeRange.lowerBound])
            if isNamespacedToolCallTag(tag) {
                return Envelope(
                    range: openRange.lowerBound..<closeRange.upperBound,
                    kind: .paired(end: "</\(tag)>")
                )
            }
            searchStart = closeRange.upperBound
        }
        return nil
    }

    private static func bracketEnvelope(in text: String) -> Envelope? {
        guard let range = MLXBracketToolCallEnvelopeScanner.range(in: text) else {
            return nil
        }
        return Envelope(range: range, kind: .consumeOnly)
    }

    private static func isNamespacedToolCallTag(_ tag: String) -> Bool {
        tag.hasSuffix(":tool_call") && tag.first?.isLetter == true
    }

    private static func partialNamespacedOpenSuffixLength(in text: String) -> Int? {
        guard let start = text.lastIndex(of: "<") else {
            return nil
        }
        let suffix = String(text[start...])
        guard isPotentialNamespacedToolCallOpenPrefix(suffix) else {
            return nil
        }
        return suffix.count
    }

    private static func isPotentialNamespacedToolCallOpenPrefix(_ text: String) -> Bool {
        guard text.hasPrefix("<"), !text.contains(">") else {
            return false
        }
        let body = text.dropFirst()
        guard body.count <= 256 else {
            return false
        }
        guard !body.isEmpty, !body.hasPrefix("/") else {
            return body.isEmpty
        }
        let parts = body.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        if parts.count == 1 {
            return isToolNamespace(parts[0])
        }
        guard parts.count == 2,
            isToolNamespace(parts[0]) else {
            return false
        }
        return "tool_call".hasPrefix(parts[1])
    }

    private static func isUnresolvedNamespacedToolCallTail(_ text: String) -> Bool {
        guard isPotentialNamespacedToolCallOpenPrefix(text) else {
            return false
        }
        let body = text.dropFirst()
        guard body.contains(":") else {
            return false
        }
        let parts = body.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        return parts.count == 2 && "tool_call".hasPrefix(parts[1])
    }

    private static func isToolNamespace(_ text: Substring) -> Bool {
        guard let first = text.first,
            first == "_" || first.isLetter else {
            return false
        }
        return text.allSatisfy { character in
            character == "_"
                || character == "."
                || character == "-"
                || character.isLetter
                || character.isNumber
        }
    }

    private static func earlier(
        _ current: Envelope?,
        _ candidate: Envelope?
    ) -> Envelope? {
        guard let candidate else {
            return current
        }
        guard let current else {
            return candidate
        }
        return candidate.range.lowerBound < current.range.lowerBound ? candidate : current
    }
}
