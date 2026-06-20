import Foundation

struct MLXCohereCommandStreamFilter {
    private static let replacements: [(marker: String, replacement: String)] = [
        ("<|START_THINKING|>", "<think>\n"),
        ("<|END_THINKING|>", "</think>\n"),
        ("<|START_TEXT|>", ""),
        ("<|END_TEXT|>", ""),
        ("<|START_RESPONSE|>", ""),
        ("<|END_RESPONSE|>", ""),
        ("<|START_OF_TURN_TOKEN|>", ""),
        ("<|END_OF_TURN_TOKEN|>", ""),
        ("<|CHATBOT_TOKEN|>", ""),
        ("<|USER_TOKEN|>", ""),
        ("<|SYSTEM_TOKEN|>", ""),
        ("<|START_TOOL_RESULT|>", ""),
        ("<|END_TOOL_RESULT|>", "")
    ]

    private var buffer = ""

    mutating func feed(_ text: String) -> String {
        guard !text.isEmpty else {
            return ""
        }
        buffer += text
        return drain(final: false)
    }

    mutating func finish() -> String {
        drain(final: true)
    }

    private mutating func drain(final: Bool) -> String {
        let keepCount = final ? 0 : partialMarkerSuffixLength(in: buffer)
        let endIndex = buffer.index(buffer.endIndex, offsetBy: -keepCount)
        let ready = String(buffer[..<endIndex])
        buffer = String(buffer[endIndex...])
        return Self.normalized(ready)
    }

    private static func normalized(_ text: String) -> String {
        var output = text
        for replacement in replacements {
            output = output.replacingOccurrences(
                of: replacement.marker,
                with: replacement.replacement
            )
        }
        return output
    }

    private func partialMarkerSuffixLength(in text: String) -> Int {
        let maximum = min(text.count, Self.longestMarkerLength - 1)
        guard maximum > 0 else {
            return 0
        }
        for length in stride(from: maximum, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if Self.replacements.contains(where: { $0.marker.hasPrefix(suffix) }) {
                return length
            }
        }
        return 0
    }

    private static var longestMarkerLength: Int {
        replacements.map(\.marker.count).max() ?? 0
    }
}
