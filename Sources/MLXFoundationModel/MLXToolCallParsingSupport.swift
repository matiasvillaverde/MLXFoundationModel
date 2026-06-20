import Foundation

enum MLXToolCallParsingSupport {
    typealias JSONObject = [String: Any]

    struct RegexMatch {
        let captures: [String]
        let range: Range<String.Index>
    }

    static func parsedCallArguments(_ text: String) -> Any {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let object = MLXToolCallLiteralParser.parseArguments(raw) {
            return object
        }
        let trimmed = raw.hasPrefix("(") && raw.hasSuffix(")")
            ? String(raw.dropFirst().dropLast())
            : raw
        if trimmed.hasPrefix("{"), let object = parsedObjectLiteral(trimmed) as? JSONObject {
            return object
        }
        let pairs = splitTopLevel(trimmed, separator: ",").compactMap(parseAssignment)
        return Dictionary(uniqueKeysWithValues: pairs)
    }

    static func parsedObjectLiteral(_ text: String) -> Any {
        if let value = MLXToolCallLiteralParser.parseValue(text) {
            return value
        }
        if let object = keyAnchoredObjectLiteral(text) {
            return object
        }
        let normalized = text.replacingOccurrences(of: #"<|"|>"#, with: #"""#)
        if let object = parseJSON(normalized) {
            return object
        }
        let singleQuoted = normalized.replacingOccurrences(of: "'", with: #"""#)
        if let object = parseJSON(singleQuoted) {
            return object
        }
        let inner = normalized.trimmingCharacters(in: CharacterSet(charactersIn: "{}"))
        let pairs = splitTopLevel(inner, separator: ",").compactMap(parseObjectPair)
        return Dictionary(uniqueKeysWithValues: pairs)
    }

    private struct KeyMatch {
        let name: String
        let start: String.Index
        let valueStart: String.Index
    }

    private static func keyAnchoredObjectLiteral(_ text: String) -> JSONObject? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.first == "{", trimmed.last == "}" else {
            return nil
        }

        let inner = String(trimmed.dropFirst().dropLast())
        let protected = markedStringsReplacedWithPlaceholders(in: inner)
        let matches = topLevelKeyMatches(in: protected.text)
        guard !matches.isEmpty else {
            return nil
        }

        var result: JSONObject = [:]
        for index in matches.indices {
            let match = matches[index]
            let valueEnd = index + 1 < matches.endIndex
                ? matches[index + 1].start
                : protected.text.endIndex
            let rawValue = String(protected.text[match.valueStart..<valueEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingTrailingComma()
            let restoredValue = restoreMarkedStringPlaceholders(
                in: rawValue,
                strings: protected.strings
            )
            result[match.name] = decodedArgumentValue(restoredValue)
        }
        return result
    }

    private static func topLevelKeyMatches(in text: String) -> [KeyMatch] {
        var matches: [KeyMatch] = []
        var state = MLXToolCallSplitState()
        var acceptsKey = true
        var index = text.startIndex
        while index < text.endIndex {
            if state.isTopLevel {
                if acceptsKey {
                    let candidate = skippingWhitespace(in: text, from: index)
                    if let match = keyMatch(in: text, from: candidate) {
                        matches.append(match)
                        acceptsKey = false
                        index = match.valueStart
                        continue
                    }
                    acceptsKey = false
                }
                if text[index] == "," {
                    acceptsKey = true
                }
            }
            state.update(with: text[index])
            index = text.index(after: index)
        }
        return matches
    }

    private static func keyMatch(in text: String, from start: String.Index) -> KeyMatch? {
        guard start < text.endIndex, isKeyStart(text[start]) else {
            return nil
        }

        var cursor = text.index(after: start)
        while cursor < text.endIndex, isKeyBody(text[cursor]) {
            cursor = text.index(after: cursor)
        }

        let separator = skippingWhitespace(in: text, from: cursor)
        guard separator < text.endIndex,
            text[separator] == ":" || text[separator] == "=" else {
            return nil
        }

        return KeyMatch(
            name: String(text[start..<cursor]),
            start: start,
            valueStart: text.index(after: separator)
        )
    }

    private static func skippingWhitespace(
        in text: String,
        from start: String.Index
    ) -> String.Index {
        var index = start
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
        return index
    }

    private static func isKeyStart(_ character: Character) -> Bool {
        character == "_" || character.isASCII && character.isLetter
    }

    private static func isKeyBody(_ character: Character) -> Bool {
        character == "_"
            || character == "."
            || character == "-"
            || character.isASCII && (character.isLetter || character.isNumber)
    }

    static func keyValuePairs(pattern: String, in text: String) -> JSONObject {
        let pairs = captureMatches(pattern: pattern, in: text, captureCount: 2).map { match in
            (match[0], decodedArgumentValue(match[1]))
        }
        return Dictionary(uniqueKeysWithValues: pairs)
    }

    static func canonicalJSONString(_ value: Any) -> String {
        if let string = value as? String,
            let parsed = parseJSON(string) {
            return canonicalJSONString(parsed)
        }
        guard
            JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let string = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return string
    }

    static func decodedArgumentValue(_ text: String) -> Any {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = trimmed.replacingOccurrences(of: #"<|"|>"#, with: #"""#)
        if let escaped = valueDelimitedBy("<escape>", in: normalized) {
            return escaped
        }
        if let value = MLXToolCallLiteralParser.parseValue(trimmed) {
            return value
        }
        if let value = parseJSON(trimmed) {
            return value
        }
        return unquoted(normalized)
    }

    private static func valueDelimitedBy(_ delimiter: String, in text: String) -> String? {
        guard text.hasPrefix(delimiter), text.hasSuffix(delimiter) else {
            return nil
        }
        return String(text.dropFirst(delimiter.count).dropLast(delimiter.count))
    }

    static func parseJSON(_ text: String) -> Any? {
        if let value = parseStrictJSON(text) {
            return value
        }
        let repaired = escapingRawControlCharacters(in: text)
        guard repaired != text else {
            return nil
        }
        return parseStrictJSON(repaired)
    }

    static func regexMatches(
        pattern: String,
        in text: String,
        captureCount: Int
    ) -> [RegexMatch] {
        guard let expression = try? NSRegularExpression(
            pattern: pattern,
            options: [.dotMatchesLineSeparators]
        ) else {
            return []
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return expression.matches(in: text, range: range).compactMap { match in
            guard let matchRange = Range(match.range, in: text) else {
                return nil
            }
            let captures = (1...captureCount).compactMap { captureIndex in
                substring(in: text, range: match.range(at: captureIndex))
            }
            return RegexMatch(captures: captures, range: matchRange)
        }
    }

    static func parseStrictJSON(_ text: String) -> Any? {
        guard let data = text.data(using: .utf8) else {
            return nil
        }
        return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    }

    static func captureMatches(
        pattern: String,
        in text: String,
        captureCount: Int
    ) -> [[String]] {
        regexMatches(pattern: pattern, in: text, captureCount: captureCount).map(\.captures)
    }

    static func blocksBetween(start: String, end: String, in text: String) -> [String] {
        var blocks: [String] = []
        var searchStart = text.startIndex
        while let startRange = text.range(of: start, range: searchStart..<text.endIndex),
            let endRange = text.range(of: end, range: startRange.upperBound..<text.endIndex) {
            blocks.append(String(text[startRange.upperBound..<endRange.lowerBound]))
            searchStart = endRange.upperBound
        }
        return blocks
    }

    static func firstJSONValueText(in text: String) -> String? {
        firstJSONValueText(in: text, opener: "{", closer: "}")
            ?? firstJSONValueText(in: text, opener: "[", closer: "]")
    }

    static func firstJSONValueText(
        in text: String,
        opener: Character,
        closer: Character
    ) -> String? {
        guard let start = text.firstIndex(of: opener) else {
            return nil
        }
        let suffix = String(text[start...])
        return balancedPrefix(in: suffix, opener: opener, closer: closer)
    }
}
