import Foundation

extension MLXToolCallLiteralParserState {
    mutating func parseValue(depth: Int) -> Any? {
        guard isWithinValueBounds(depth: depth) else {
            return nil
        }

        skipWhitespace()
        if starts(with: #"<|"|>"#) {
            return parseMarkedString()
        }

        return parseValueAfterWhitespace(depth: depth)
    }

    mutating func parseKey() -> String? {
        skipWhitespace()
        if starts(with: #"<|"|>"#) {
            return parseMarkedString()
        }
        if peek == "\"" {
            return parseDoubleQuotedString()
        }
        if peek == "'" {
            return parseSingleQuotedString()
        }

        return parseBareKey()
    }

    private mutating func parseValueAfterWhitespace(depth: Int) -> Any? {
        switch peek {
        case "\"":
            return parseDoubleQuotedString()

        case "'":
            return parseSingleQuotedString()

        case "{":
            return parseObject(open: "{", close: "}", depth: depth + 1)

        case "[":
            return parseArray(depth: depth + 1)

        case "(":
            return parseObject(open: "(", close: ")", depth: depth + 1)

        default:
            return parseBareValue()
        }
    }

    private mutating func parseBareKey() -> String? {
        let start = index
        while let character = peek, character != ":", character != "=" {
            if Self.bareKeyTerminators.contains(character) {
                return nil
            }
            advance()
        }
        let key = String(text[start..<index]).trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }

    private mutating func parseBareValue() -> Any? {
        let start = index
        while let character = peek, !Self.bareValueTerminators.contains(character) {
            advance()
        }
        let value = String(text[start..<index]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            return nil
        }
        return parsedBareValue(value)
    }

    private func parsedBareValue(_ value: String) -> Any {
        if value.hasPrefix("<escape>"), value.hasSuffix("<escape>") {
            return String(value.dropFirst("<escape>".count).dropLast("<escape>".count))
        }
        switch value {
        case "True":
            return true

        case "False":
            return false

        case "None":
            return NSNull()

        default:
            break
        }
        if let parsed = MLXToolCallParsingSupport.parseJSON(value) {
            return parsed
        }
        if let intValue = Int(value) {
            return intValue
        }
        if let doubleValue = Double(value) {
            return doubleValue
        }
        return value
    }

    private func isWithinValueBounds(depth: Int) -> Bool {
        text.utf8.count <= MLXToolCallLiteralParser.maximumLength
            && depth <= MLXToolCallLiteralParser.maximumDepth
    }

    private static let bareKeyTerminators: Set<Character> = [",", "}", "]", ")"]
    private static let bareValueTerminators: Set<Character> = [",", "}", "]", ")"]
}
