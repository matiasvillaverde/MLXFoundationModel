import Foundation

struct MLXMiniMaxM3StreamFilter {
    private struct Replacement {
        let marker: String
        let replacement: String
    }

    private static let replacements: [Replacement] = [
        Replacement(marker: "<mm:think>", replacement: "<think>"),
        Replacement(marker: "</mm:think>", replacement: "</think>"),
        Replacement(marker: "[e~[", replacement: ""),
        Replacement(marker: "]~b]", replacement: ""),
        Replacement(marker: "]~!b[", replacement: ""),
        Replacement(marker: "]!p~[", replacement: ""),
        Replacement(marker: "]!d~[", replacement: "")
    ]

    private static let markers = replacements.map(\.marker)

    private var buffer = ""
    private var shouldDropLeadingReplacement = false

    mutating func feed(_ text: String) -> String {
        guard !text.isEmpty else {
            return ""
        }

        buffer += String.consumingLeadingUnicodeReplacementIfNeeded(
            from: text,
            shouldDrop: &shouldDropLeadingReplacement
        )
        let retainCount = partialMarkerSuffixLength(in: buffer)
        guard retainCount > 0 else {
            defer {
                shouldDropLeadingReplacement = Self.endsWithMarker(buffer)
                buffer = ""
            }
            return Self.replaceMarkers(in: buffer)
        }

        let split = buffer.index(buffer.endIndex, offsetBy: -retainCount)
        let ready = String(buffer[..<split])
            .droppingTrailingUnicodeReplacementCharacter()
        buffer = String(buffer[split...])
        shouldDropLeadingReplacement = Self.endsWithMarker(ready)
        return Self.replaceMarkers(in: ready)
    }

    mutating func finish() -> String {
        defer {
            buffer = ""
        }
        return Self.replaceMarkers(in: buffer)
    }

    private static func replaceMarkers(in text: String) -> String {
        replacements.reduce(text) { partial, replacement in
            partial
                .replacingOccurrences(
                    of: "\u{FFFD}\(replacement.marker)",
                    with: replacement.replacement
                )
                .replacingOccurrences(
                    of: "\(replacement.marker)\u{FFFD}",
                    with: replacement.replacement
                )
                .replacingOccurrences(
                    of: replacement.marker,
                    with: replacement.replacement
                )
        }
    }

    private func partialMarkerSuffixLength(in text: String) -> Int {
        let longestMarkerLength = Self.markers.map(\.count).max() ?? 0
        let maximum = max(0, min(text.count, longestMarkerLength - 1))
        guard maximum > 0 else {
            return 0
        }

        for length in stride(from: maximum, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if Self.markers.contains(where: { $0.hasPrefix(suffix) }) {
                return length
            }
        }

        return 0
    }
    private static func endsWithMarker(_ text: String) -> Bool {
        replacements.contains { replacement in
            text.hasSuffix(replacement.marker)
        }
    }
}
