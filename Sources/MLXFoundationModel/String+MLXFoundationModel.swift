import Foundation

extension String {
    func droppingJSONValueEnvelope() -> String {
        let prefix = #"{"value":"#
        guard hasPrefix(prefix), hasSuffix("}") else {
            return self
        }
        return String(dropFirst(prefix.count).dropLast())
    }

    func suffixString(retainingAtMost count: Int) -> String {
        guard count > 0, self.count > count else {
            return self
        }
        let start = index(endIndex, offsetBy: -count)
        return String(self[start...])
    }

    func trimmingOneBoundaryNewline() -> String {
        var text = self
        if text.first == "\n" {
            text.removeFirst()
        }
        if text.last == "\n" {
            text.removeLast()
        }
        return text
    }

    func droppingLeadingUnicodeReplacementCharacter() -> String {
        guard first == "\u{FFFD}" else {
            return self
        }
        return String(dropFirst())
    }

    func droppingTrailingUnicodeReplacementCharacter() -> String {
        guard last == "\u{FFFD}" else {
            return self
        }
        return String(dropLast())
    }

    static func consumingLeadingUnicodeReplacementIfNeeded(
        from text: String,
        shouldDrop: inout Bool
    ) -> String {
        guard shouldDrop else {
            return text
        }
        shouldDrop = false
        return text.droppingLeadingUnicodeReplacementCharacter()
    }

    static func dropLeadingUnicodeReplacement(
        from buffer: inout String,
        orNextChunk shouldDrop: inout Bool
    ) {
        let stripped = buffer.droppingLeadingUnicodeReplacementCharacter()
        if stripped.count != buffer.count {
            buffer = stripped
            shouldDrop = false
        } else {
            shouldDrop = buffer.isEmpty
        }
    }
}
