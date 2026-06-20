import Foundation

struct MLXToolCallLiteralParserState {
    var index: String.Index
    let text: String

    init(_ text: String) {
        self.text = text
        self.index = text.startIndex
    }

    var isAtEnd: Bool {
        index >= text.endIndex
    }

    var peek: Character? {
        isAtEnd ? nil : text[index]
    }

    mutating func advance() {
        index = text.index(after: index)
    }

    mutating func consume(_ character: Character) -> Bool {
        guard peek == character else {
            return false
        }
        advance()
        return true
    }

    func starts(with prefix: String) -> Bool {
        text[index...].hasPrefix(prefix)
    }

    mutating func skipWhitespace() {
        while let character = peek, character.isWhitespace {
            advance()
        }
    }
}
