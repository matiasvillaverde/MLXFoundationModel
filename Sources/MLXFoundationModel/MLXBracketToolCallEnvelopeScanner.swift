import Foundation

enum MLXBracketToolCallEnvelopeScanner {
    private static let prefixes = ["[Calling tool:", "[Tool call:"]

    static func range(in text: String) -> Range<String.Index>? {
        prefixes.reduce(nil) { result, prefix in
            guard let range = firstRange(in: text, prefix: prefix) else {
                return result
            }
            guard let result else {
                return range
            }
            return range.lowerBound < result.lowerBound ? range : result
        }
    }

    static func partialSuffixLength(in text: String) -> Int? {
        let partialPrefixLength = prefixes
            .compactMap { partialPrefixSuffixLength(in: text, prefix: $0) }
            .max()
        let candidateLength = incompleteCandidateSuffixLength(in: text)
        return [partialPrefixLength, candidateLength]
            .compactMap(\.self)
            .max()
    }

    static func shouldDropUnresolvedTail(_ text: String) -> Bool {
        incompleteCandidateSuffixLength(in: text) == text.count
    }

    private static func firstRange(
        in text: String,
        prefix: String
    ) -> Range<String.Index>? {
        var searchStart = text.startIndex
        while let prefixRange = text.range(of: prefix, range: searchStart..<text.endIndex) {
            if let range = parseCompleteRange(in: text, prefixRange: prefixRange) {
                return range
            }
            searchStart = text.index(after: prefixRange.lowerBound)
        }
        return nil
    }

    private static func parseCompleteRange(
        in text: String,
        prefixRange: Range<String.Index>
    ) -> Range<String.Index>? {
        var cursor = skippingWhitespace(in: text, from: prefixRange.upperBound)
        guard parseName(in: text, cursor: &cursor) else {
            return nil
        }
        cursor = skippingWhitespace(in: text, from: cursor)
        if cursor < text.endIndex, text[cursor] == "(" {
            guard let argumentRange = balancedArgumentRange(in: text, from: cursor) else {
                return nil
            }
            cursor = skippingWhitespace(in: text, from: argumentRange.upperBound)
        }
        guard cursor < text.endIndex, text[cursor] == "]" else {
            return nil
        }
        return prefixRange.lowerBound..<text.index(after: cursor)
    }

    private static func balancedArgumentRange(
        in text: String,
        from cursor: String.Index
    ) -> Range<String.Index>? {
        let scanner = MLXBalancedPrefixScanner(text: text)
        guard let argumentText = scanner.scan(from: cursor, opener: "(", closer: ")") else {
            return nil
        }
        return cursor..<text.index(cursor, offsetBy: argumentText.count)
    }

    private static func incompleteCandidateSuffixLength(in text: String) -> Int? {
        guard let prefixRange = lastPrefixRange(in: text) else {
            return nil
        }
        let suffix = String(text[prefixRange.lowerBound...])
        guard isIncompleteCandidate(suffix, prefixLength: text.distance(
            from: prefixRange.lowerBound,
            to: prefixRange.upperBound
        )) else {
            return nil
        }
        return suffix.count
    }

    private static func lastPrefixRange(in text: String) -> Range<String.Index>? {
        prefixes.compactMap { prefix in
            text.range(of: prefix, options: .backwards)
        }
        .max { $0.lowerBound < $1.lowerBound }
    }

    private static func isIncompleteCandidate(
        _ text: String,
        prefixLength: Int
    ) -> Bool {
        guard !text.contains("]") else {
            return false
        }
        var cursor = text.index(text.startIndex, offsetBy: prefixLength)
        cursor = skippingWhitespace(in: text, from: cursor)
        guard cursor < text.endIndex else {
            return true
        }
        guard isNameStart(text[cursor]) else {
            return false
        }
        _ = parseName(in: text, cursor: &cursor)
        cursor = skippingWhitespace(in: text, from: cursor)
        guard cursor < text.endIndex else {
            return true
        }
        return text[cursor] == "("
    }

    private static func partialPrefixSuffixLength(
        in text: String,
        prefix: String
    ) -> Int? {
        let maximum = min(text.count, prefix.count - 1)
        guard maximum > 0 else {
            return nil
        }
        for length in stride(from: maximum, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if prefix.hasPrefix(suffix) {
                return length
            }
        }
        return nil
    }

    private static func parseName(
        in text: String,
        cursor: inout String.Index
    ) -> Bool {
        guard cursor < text.endIndex, isNameStart(text[cursor]) else {
            return false
        }
        cursor = text.index(after: cursor)
        while cursor < text.endIndex, isNameBody(text[cursor]) {
            cursor = text.index(after: cursor)
        }
        return true
    }

    private static func skippingWhitespace(
        in text: String,
        from cursor: String.Index
    ) -> String.Index {
        var cursor = cursor
        while cursor < text.endIndex, text[cursor].isWhitespace {
            cursor = text.index(after: cursor)
        }
        return cursor
    }

    private static func isNameStart(_ character: Character) -> Bool {
        character == "_" || character.isASCII && character.isLetter
    }

    private static func isNameBody(_ character: Character) -> Bool {
        character == "_"
            || character == "."
            || character == "-"
            || character.isASCII && (character.isLetter || character.isNumber)
    }
}
