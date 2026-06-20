import Foundation

enum MLXGemma4HistorySanitizer {
    private static let thinkOpen = "<think>"
    private static let thinkClose = "</think>"
    private static let channelOpen = "<|channel>"
    private static let channelClose = "<channel|>"
    private static let strayToolMarkers = [
        "<|tool_call>",
        "<tool_call|>"
    ]

    static func sanitize(_ content: String) -> String {
        strayToolMarkers.reduce(stripLeadingThinking(from: content)) { partial, marker in
            partial.replacingOccurrences(of: marker, with: "")
        }
    }

    private static func stripLeadingThinking(from content: String) -> String {
        var result = content
        var stripped = false

        while let strippedResult = result.strippingOneLeadingGemmaThinkingBlock(
            thinkOpen: thinkOpen,
            thinkClose: thinkClose,
            channelOpen: channelOpen,
            channelClose: channelClose
        ) {
            result = strippedResult.trimmingLeadingWhitespace()
            stripped = true
        }

        return stripped ? result : content
    }
}

extension String {
    func strippingOneLeadingGemmaThinkingBlock(
        thinkOpen: String,
        thinkClose: String,
        channelOpen: String,
        channelClose: String
    ) -> String? {
        let leadingRange = firstNonWhitespaceIndex() ?? endIndex
        let suffix = self[leadingRange...]
        if suffix.hasPrefix(thinkOpen) {
            return stripLeadingBlock(
                from: leadingRange,
                closeMarker: thinkClose
            )
        }
        if suffix.hasPrefix(channelOpen) {
            return stripLeadingBlock(
                from: leadingRange,
                closeMarker: channelClose
            )
        }
        return nil
    }

    func trimmingLeadingWhitespace() -> String {
        guard let index = firstNonWhitespaceIndex() else {
            return ""
        }
        return String(self[index...])
    }

    private func stripLeadingBlock(
        from start: String.Index,
        closeMarker: String
    ) -> String? {
        guard let closeRange = range(of: closeMarker, range: start..<endIndex) else {
            return nil
        }
        return String(self[closeRange.upperBound...])
    }

    private func firstNonWhitespaceIndex() -> String.Index? {
        firstIndex { !$0.isWhitespace }
    }
}
