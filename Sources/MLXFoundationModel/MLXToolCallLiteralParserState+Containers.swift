import Foundation

extension MLXToolCallLiteralParserState {
    mutating func parseObject(
        open: Character,
        close: Character,
        depth: Int
    ) -> Any? {
        guard consume(open) else {
            return nil
        }
        return parseEntries(close: close, depth: depth + 1)
    }

    mutating func parseEntries(
        close: Character?,
        depth: Int
    ) -> Any? {
        guard isWithinBounds(depth: depth) else {
            return nil
        }

        var result: [String: Any] = [:]
        skipWhitespace()
        if let close, consume(close) {
            return result
        }

        while !isAtEnd {
            if let close, consume(close) {
                return result
            }
            guard parseEntry(into: &result, depth: depth) else {
                return nil
            }
            switch consumeEntryDelimiter(close: close) {
            case .continueParsing:
                continue

            case .finished:
                return result

            case .invalid:
                return nil
            }
        }

        return close == nil ? result : nil
    }

    mutating func parseArray(depth: Int) -> Any? {
        guard consume("[") else {
            return nil
        }

        var values: [Any] = []
        skipWhitespace()
        if consume("]") {
            return values
        }

        while !isAtEnd {
            guard let value = parseValue(depth: depth + 1) else {
                return nil
            }
            values.append(value)
            if consumeArrayDelimiter() {
                return values
            }
        }

        return nil
    }

    private mutating func parseEntry(
        into result: inout [String: Any],
        depth: Int
    ) -> Bool {
        guard let key = parseKey() else {
            return false
        }
        skipWhitespace()
        guard consume(":") || consume("="),
            let value = parseValue(depth: depth + 1) else {
            return false
        }
        result[key] = value
        return true
    }

    private mutating func consumeArrayDelimiter() -> Bool {
        skipWhitespace()
        if consume(",") {
            skipWhitespace()
            return consume("]")
        }
        return consume("]")
    }

    private mutating func consumeEntryDelimiter(close: Character?) -> EntryDelimiter {
        skipWhitespace()
        if consume(",") {
            skipWhitespace()
            if let close, consume(close) {
                return .finished
            }
            return .continueParsing
        }
        guard let close else {
            return isAtEnd ? .finished : .invalid
        }
        return consume(close) ? .finished : .invalid
    }

    private enum EntryDelimiter {
        case continueParsing
        case finished
        case invalid
    }

    private func isWithinBounds(depth: Int) -> Bool {
        text.utf8.count <= MLXToolCallLiteralParser.maximumLength
            && depth <= MLXToolCallLiteralParser.maximumDepth
    }
}
