import Foundation

extension MLXBalancedPrefixScanner {
    func skippedStringEnd(from index: String.Index) -> String.Index? {
        if text[index...].hasPrefix(#"<|"|>"#) {
            let start = text.index(index, offsetBy: #"<|"|>"#.count)
            return text.range(of: #"<|"|>"#, range: start..<text.endIndex)?.upperBound
        }
        if text[index] == "\"" {
            return doubleQuotedStringEnd(from: index)
        }
        if text[index] == "'" {
            return singleQuotedStringEnd(from: index)
        }
        return nil
    }

    func doubleQuotedStringEnd(from startIndex: String.Index) -> String.Index? {
        var index = text.index(after: startIndex)
        var isEscaped = false
        while index < text.endIndex {
            let character = text[index]
            if isEscaped {
                isEscaped = false
            } else if character == "\\" {
                isEscaped = true
            } else if character == "\"" {
                return text.index(after: index)
            }
            index = text.index(after: index)
        }
        return nil
    }

    func singleQuotedStringEnd(from startIndex: String.Index) -> String.Index? {
        var index = text.index(after: startIndex)
        var isEscaped = false
        while index < text.endIndex {
            let character = text[index]
            if isEscaped {
                isEscaped = false
            } else if character == "\\" {
                isEscaped = true
            } else if character == "'", singleQuoteCanClose(at: index) {
                return text.index(after: index)
            }
            index = text.index(after: index)
        }
        return nil
    }

    func singleQuoteCanClose(at quoteIndex: String.Index) -> Bool {
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

    private static let singleQuoteCloseAnchors: Set<Character> = [",", "}", "]", ")"]
}
