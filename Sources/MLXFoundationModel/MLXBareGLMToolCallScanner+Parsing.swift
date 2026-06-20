import Foundation

extension MLXBareGLMToolCallScanner {
    private struct ParsedArgument {
        let key: String
        let value: Any
        let valueEnd: String.Index
    }

    private enum Continuation {
        case nextPair(String.Index)
        case complete(String.Index)
        case incomplete
    }

    static func firstParsedCall(
        in text: String,
        searchRange: Range<String.Index>,
        scope: ToolNameScope,
        final: Bool
    ) -> ParsedCall? {
        var cursor = searchRange.lowerBound
        while cursor < searchRange.upperBound {
            guard let start = startRange(
                in: text,
                searchRange: cursor..<searchRange.upperBound,
                scope: scope
            ) else {
                return nil
            }
            if let parsed = parsedCall(in: text, start: start.lowerBound, scope: scope, final: final) {
                return parsed
            }
            cursor = start.upperBound
        }
        return nil
    }

    static func startRange(
        in text: String,
        searchRange: Range<String.Index>,
        scope: ToolNameScope
    ) -> Range<String.Index>? {
        switch scope {
        case .any:
            return firstGenericStart(in: text, searchRange: searchRange)

        case .known(let toolNames):
            return firstKnownStart(in: text, searchRange: searchRange, toolNames: toolNames)
        }
    }

    private static func firstGenericStart(
        in text: String,
        searchRange: Range<String.Index>
    ) -> Range<String.Index>? {
        var cursor = searchRange.lowerBound
        while let keyRange = text.range(of: keyStart, range: cursor..<searchRange.upperBound) {
            let nameStart = toolNameStart(before: keyRange.lowerBound, in: text)
            if isValidGenericStart(nameStart..<keyRange.lowerBound, in: text) {
                return nameStart..<keyRange.upperBound
            }
            cursor = keyRange.upperBound
        }
        return nil
    }

    private static func isValidGenericStart(
        _ range: Range<String.Index>,
        in text: String
    ) -> Bool {
        guard range.lowerBound < range.upperBound,
            hasBoundaryBefore(range.lowerBound, in: text) else {
            return false
        }
        return isValidToolName(String(text[range]))
    }

    private static func parsedCall(
        in text: String,
        start: String.Index,
        scope: ToolNameScope,
        final: Bool
    ) -> ParsedCall? {
        guard let nameRange = toolNameRange(in: text, start: start, scope: scope),
            text[nameRange.upperBound...].hasPrefix(keyStart) else {
            return nil
        }

        let name = String(text[nameRange])
        var cursor = nameRange.upperBound
        var arguments: [String: Any] = [:]

        while cursor < text.endIndex {
            guard let parsed = parsedArgument(in: text, cursor: cursor) else {
                return nil
            }
            arguments[parsed.key] = parsed.value

            switch continuation(after: parsed.valueEnd, in: text, final: final) {
            case .nextPair(let nextCursor):
                cursor = nextCursor

            case .complete(let end):
                return parsedCall(name: name, arguments: arguments, start: start, end: end)

            case .incomplete:
                return nil
            }
        }
        return nil
    }

    private static func parsedArgument(
        in text: String,
        cursor: String.Index
    ) -> ParsedArgument? {
        guard text[cursor...].hasPrefix(keyStart),
            let keyClose = text.range(of: keyEnd, range: cursor..<text.endIndex) else {
            return nil
        }
        let keyStartIndex = text.index(cursor, offsetBy: keyStart.count)
        let key = String(text[keyStartIndex..<keyClose.lowerBound])
        var valueCursor = skipWhitespace(from: keyClose.upperBound, in: text)
        guard text[valueCursor...].hasPrefix(valueStart) else {
            return nil
        }
        valueCursor = text.index(valueCursor, offsetBy: valueStart.count)
        guard let valueClose = text.range(of: valueEnd, range: valueCursor..<text.endIndex) else {
            return nil
        }
        let valueText = String(text[valueCursor..<valueClose.lowerBound])
        return ParsedArgument(
            key: key,
            value: MLXToolCallParsingSupport.decodedArgumentValue(valueText),
            valueEnd: valueClose.upperBound
        )
    }

    private static func continuation(
        after valueEnd: String.Index,
        in text: String,
        final: Bool
    ) -> Continuation {
        let nextCursor = skipWhitespace(from: valueEnd, in: text)
        guard nextCursor < text.endIndex else {
            return final ? .complete(valueEnd) : .incomplete
        }
        let tail = String(text[nextCursor...])
        if tail.hasPrefix(keyStart) {
            return .nextPair(nextCursor)
        }
        if keyStart.hasPrefix(tail) {
            return .incomplete
        }
        return .complete(valueEnd)
    }

    private static func parsedCall(
        name: String,
        arguments: [String: Any],
        start: String.Index,
        end: String.Index
    ) -> ParsedCall {
        ParsedCall(
            range: start..<end,
            call: MLXExtractedToolCall(
                name: name,
                argumentsJSON: MLXToolCallParsingSupport.canonicalArgumentsJSONString(arguments)
            )
        )
    }

    private static func toolNameRange(
        in text: String,
        start: String.Index,
        scope: ToolNameScope
    ) -> Range<String.Index>? {
        switch scope {
        case .any:
            guard let keyRange = text.range(of: keyStart, range: start..<text.endIndex),
                start < keyRange.lowerBound else {
                return nil
            }
            return start..<keyRange.lowerBound

        case .known:
            for name in scope.sortedNames where text[start...].hasPrefix(name) {
                let end = text.index(start, offsetBy: name.count)
                return start..<end
            }
            return nil
        }
    }

    private static func toolNameStart(
        before index: String.Index,
        in text: String
    ) -> String.Index {
        var cursor = index
        while cursor > text.startIndex {
            let previous = text.index(before: cursor)
            guard isToolNameBody(text[previous]) else {
                break
            }
            cursor = previous
        }
        return cursor
    }

    private static func skipWhitespace(
        from index: String.Index,
        in text: String
    ) -> String.Index {
        var cursor = index
        while cursor < text.endIndex, text[cursor].isWhitespace {
            cursor = text.index(after: cursor)
        }
        return cursor
    }

    private static func isValidToolName(_ name: String) -> Bool {
        guard let first = name.first,
            first == "_" || first.isLetter else {
            return false
        }
        return name.allSatisfy(isToolNameBody)
    }
}
