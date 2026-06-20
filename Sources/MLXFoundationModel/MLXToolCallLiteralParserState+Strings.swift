import Foundation

extension MLXToolCallLiteralParserState {
    mutating func parseMarkedString() -> String? {
        let delimiter = #"<|"|>"#
        guard starts(with: delimiter) else {
            return nil
        }
        let start = text.index(index, offsetBy: delimiter.count)
        guard let end = text.range(of: delimiter, range: start..<text.endIndex)?.lowerBound else {
            return nil
        }
        index = text.index(end, offsetBy: delimiter.count)
        return String(text[start..<end])
    }

    mutating func parseDoubleQuotedString() -> String? {
        let start = index
        advance()
        var isEscaped = false
        while let character = peek {
            if isEscaped {
                isEscaped = false
                advance()
                continue
            }
            if character == "\\" {
                isEscaped = true
                advance()
                continue
            }
            if character == "\"" {
                advance()
                let token = String(text[start..<index])
                return MLXToolCallParsingSupport.parseJSON(token) as? String
            }
            advance()
        }
        return nil
    }

    mutating func parseSingleQuotedString() -> String? {
        guard consume("'") else {
            return nil
        }

        var result = ""
        var isEscaped = false
        while let character = peek {
            if consumeEscaped(character, isEscaped: &isEscaped, result: &result) {
                continue
            }
            if character == "'", singleQuoteCanClose(at: index) {
                advance()
                return result
            }
            result.append(character)
            advance()
        }

        return nil
    }

    private mutating func consumeEscaped(
        _ character: Character,
        isEscaped: inout Bool,
        result: inout String
    ) -> Bool {
        if isEscaped {
            result.append(character)
            isEscaped = false
            advance()
            return true
        }
        if character == "\\" {
            isEscaped = true
            advance()
            return true
        }
        return false
    }

    private func singleQuoteCanClose(at quoteIndex: String.Index) -> Bool {
        var cursor = text.index(after: quoteIndex)
        while cursor < text.endIndex {
            let character = text[cursor]
            if character.isWhitespace {
                cursor = text.index(after: cursor)
                continue
            }
            return Self.singleQuoteCloseAnchors.contains(character)
        }
        return true
    }

    private static let singleQuoteCloseAnchors: Set<Character> = [":", "=", ",", "}", "]", ")"]
}
