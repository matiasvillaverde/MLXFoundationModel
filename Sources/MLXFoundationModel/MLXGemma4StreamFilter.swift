import Foundation

struct MLXGemma4StreamFilter {
    private struct MarkerMatch {
        let range: Range<String.Index>
        let marker: String
    }

    private static let thoughtOpen = "<|channel>thought\n"
    private static let bareChannelOpen = "<|channel>"
    private static let thoughtClose = "<channel|>"
    private static let turnEnd = "<turn|>"
    private static let toolResponseOpen = "<|tool_response>"
    private static let toolResponseClose = "<tool_response|>"
    private static let thinkOpen = "<think>\n"
    private static let thinkClose = "</think>\n"
    private static let markers = [
        thoughtOpen,
        bareChannelOpen,
        thoughtClose,
        turnEnd,
        toolResponseOpen,
        toolResponseClose
    ]

    private var buffer = ""
    private var inThought = false
    private var shouldDropLeadingReplacement = false

    mutating func feed(_ text: String) -> String {
        buffer += String.consumingLeadingUnicodeReplacementIfNeeded(
            from: text,
            shouldDrop: &shouldDropLeadingReplacement
        )
        return drain(final: false)
    }

    mutating func finish() -> String {
        drain(final: true)
    }

    private mutating func drain(final: Bool) -> String {
        var output = ""
        while !buffer.isEmpty {
            guard let match = firstMarker(in: buffer) else {
                output += drainUnmarkedRemainder(final: final)
                break
            }

            if shouldDeferBareOpen(match: match, final: final) {
                output += String(buffer[..<match.range.lowerBound])
                buffer = String(buffer[match.range.lowerBound...])
                break
            }

            output += String(buffer[..<match.range.lowerBound])
                .droppingTrailingUnicodeReplacementCharacter()
            buffer = String(buffer[match.range.upperBound...])
            String.dropLeadingUnicodeReplacement(
                from: &buffer,
                orNextChunk: &shouldDropLeadingReplacement
            )
            output += consume(marker: match.marker)
        }

        if final, inThought {
            inThought = false
            output += Self.thinkClose
        }
        return output
    }

    private mutating func drainUnmarkedRemainder(final: Bool) -> String {
        guard !final else {
            defer { buffer = "" }
            return buffer
        }
        let retainCount = partialMarkerSuffixLength(in: buffer)
        guard buffer.count > retainCount else {
            return ""
        }
        let split = retainCount == 0
            ? buffer.endIndex
            : buffer.index(buffer.endIndex, offsetBy: -retainCount)
        var visible = String(buffer[..<split])
        if retainCount > 0 {
            visible = visible.droppingTrailingUnicodeReplacementCharacter()
        }
        buffer = String(buffer[split...])
        return visible
    }

    private mutating func consume(marker: String) -> String {
        switch marker {
        case Self.thoughtOpen:
            return openThoughtIfNeeded()

        case Self.bareChannelOpen:
            return openMalformedThought()

        case Self.thoughtClose:
            return closeThoughtIfNeeded()

        default:
            return ""
        }
    }

    private mutating func openThoughtIfNeeded() -> String {
        guard !inThought else {
            return ""
        }
        inThought = true
        return Self.thinkOpen
    }

    private mutating func openMalformedThought() -> String {
        let output = openThoughtIfNeeded()
        if buffer.hasPrefix("thought\n") {
            buffer.removeFirst("thought\n".count)
        } else if buffer.hasPrefix("thought") {
            buffer.removeFirst("thought".count)
        }
        return output
    }

    private mutating func closeThoughtIfNeeded() -> String {
        guard inThought else {
            return ""
        }
        inThought = false
        return Self.thinkClose
    }

    private func shouldDeferBareOpen(match: MarkerMatch, final: Bool) -> Bool {
        guard !final, match.marker == Self.bareChannelOpen else {
            return false
        }
        let suffix = String(buffer[match.range.lowerBound...])
        return suffix.count < Self.thoughtOpen.count && Self.thoughtOpen.hasPrefix(suffix)
    }

    private func firstMarker(in text: String) -> MarkerMatch? {
        Self.markers
            .compactMap { marker -> MarkerMatch? in
                guard let range = text.range(of: marker) else {
                    return nil
                }
                return MarkerMatch(range: range, marker: marker)
            }
            .min { lhs, rhs in
                if lhs.range.lowerBound == rhs.range.lowerBound {
                    return lhs.marker.count > rhs.marker.count
                }
                return lhs.range.lowerBound < rhs.range.lowerBound
            }
    }

    private func partialMarkerSuffixLength(in text: String) -> Int {
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

    private var longestMarkerLength: Int {
        Self.markers.map(\.count).max() ?? 0
    }
}
