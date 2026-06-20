import Foundation

extension MLXToolCallParsingSupport {
    struct PlaceholderText {
        let text: String
        let strings: [String]
    }

    static func markedStringsReplacedWithPlaceholders(in text: String) -> PlaceholderText {
        let marker = #"<|"|>"#
        var strings: [String] = []
        var output = ""
        var searchStart = text.startIndex

        while let startRange = text.range(of: marker, range: searchStart..<text.endIndex),
            let endRange = text.range(of: marker, range: startRange.upperBound..<text.endIndex) {
            output += text[searchStart..<startRange.lowerBound]
            let value = String(text[startRange.upperBound..<endRange.lowerBound])
            strings.append(value)
            output += placeholder(for: strings.count - 1)
            searchStart = endRange.upperBound
        }

        output += text[searchStart...]
        return PlaceholderText(text: output, strings: strings)
    }

    static func restoreMarkedStringPlaceholders(
        in text: String,
        strings: [String]
    ) -> String {
        strings.indices.reduce(text) { result, index in
            result.replacingOccurrences(of: placeholder(for: index), with: strings[index])
        }
    }

    private static func placeholder(for index: Int) -> String {
        "\u{0}\(index)\u{0}"
    }

    static func parseObjectPair(_ text: String) -> (String, Any)? {
        parsePair(text, separators: [":", "="])
    }

    static func parseAssignment(_ text: String) -> (String, Any)? {
        parsePair(text, separators: ["=", ":"])
    }

    static func parsePair(_ text: String, separators: [Character]) -> (String, Any)? {
        for separator in separators {
            guard let index = firstTopLevelSeparator(separator, in: text) else {
                continue
            }
            let key = unquoted(String(text[..<index]))
            let value = String(text[text.index(after: index)...])
            guard !key.isEmpty else {
                return nil
            }
            return (key, decodedArgumentValue(value))
        }
        return nil
    }

    static func substring(in text: String, range: NSRange) -> String? {
        guard range.location != NSNotFound, let stringRange = Range(range, in: text) else {
            return nil
        }
        return String(text[stringRange])
    }

    static func escapingRawControlCharacters(in text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        var inString = false
        var isEscaped = false

        for scalar in text.unicodeScalars {
            if isEscaped {
                result.unicodeScalars.append(scalar)
                isEscaped = false
                continue
            }
            if scalar == "\\" {
                result.unicodeScalars.append(scalar)
                isEscaped = true
                continue
            }
            if scalar == "\"" {
                inString.toggle()
                result.unicodeScalars.append(scalar)
                continue
            }
            if inString, scalar.value < 0x20 {
                result += escapedControlScalar(scalar)
            } else {
                result.unicodeScalars.append(scalar)
            }
        }

        return result
    }

    private static func escapedControlScalar(_ scalar: Unicode.Scalar) -> String {
        switch scalar {
        case "\u{08}":
            return #"\b"#

        case "\u{09}":
            return #"\t"#

        case "\u{0A}":
            return #"\n"#

        case "\u{0C}":
            return #"\f"#

        case "\u{0D}":
            return #"\r"#

        default:
            return String(format: #"\u%04X"#, scalar.value)
        }
    }

    static func splitTopLevel(_ text: String, separator: Character) -> [String] {
        var parts: [String] = []
        var start = text.startIndex
        var state = MLXToolCallSplitState()
        for index in text.indices {
            state.update(with: text[index])
            if text[index] == separator, state.isTopLevel {
                parts.append(String(text[start..<index]))
                start = text.index(after: index)
            }
        }
        parts.append(String(text[start...]))
        return parts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    static func firstTopLevelSeparator(
        _ separator: Character,
        in text: String
    ) -> String.Index? {
        var state = MLXToolCallSplitState()
        for index in text.indices {
            state.update(with: text[index])
            if text[index] == separator, state.isTopLevel {
                return index
            }
        }
        return nil
    }

    static func unquoted(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let quotes: [(Character, Character)] = [("\"", "\""), ("'", "'")]
        for (open, close) in quotes where trimmed.first == open && trimmed.last == close {
            return String(trimmed.dropFirst().dropLast())
        }
        return trimmed
    }

    struct StringScanState {
        private var inString = false
        private var delimiter: Character?
        private var isEscaped = false

        mutating func update(with character: Character) -> Bool {
            if isEscaped {
                isEscaped = false
                return inString
            }
            if character == "\\" {
                isEscaped = true
                return inString
            }
            if inString, character == delimiter {
                inString = false
                delimiter = nil
                return true
            }
            if !inString, character == "\"" || character == "'" {
                inString = true
                delimiter = character
                return true
            }
            return inString
        }
    }
}

extension String {
    func trimmingTrailingComma() -> String {
        var result = self
        while let last = result.last, last.isWhitespace || last == "," {
            result.removeLast()
        }
        return result
    }
}
