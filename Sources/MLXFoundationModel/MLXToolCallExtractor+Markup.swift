import Foundation

extension MLXToolCallExtractor {
    static func extractXMLToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let blocks = Parser.captureMatches(
            pattern: #"<tool_call>(.*?)</tool_call>"#,
            in: text,
            captureCount: 1
        )
        var calls: [MLXExtractedToolCall] = []
        for block in blocks.compactMap(\.first) {
            let jsonCalls = extractJSONToolCalls(from: block.trimmingCharacters(in: .whitespacesAndNewlines))
            if !jsonCalls.isEmpty {
                calls.append(contentsOf: jsonCalls)
                continue
            }
            if let call = extractFunctionXMLCall(from: block) {
                calls.append(call)
                continue
            }
            if let call = extractGLMXMLCall(from: block) {
                calls.append(call)
                continue
            }
            if let call = extractPlainTextXMLCall(from: block) {
                calls.append(call)
            }
        }
        return calls
    }

    private static func extractFunctionXMLCall(from text: String) -> MLXExtractedToolCall? {
        guard let match = Parser.captureMatches(
            pattern: #"<function=([^>\s]+)>(.*?)</function>"#,
            in: text,
            captureCount: 2
        ).first else {
            return nil
        }
        let parameters = Parser.keyValuePairs(
            pattern: #"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>"#,
            in: match[1]
        )
        return MLXExtractedToolCall(
            name: match[0],
            argumentsJSON: Parser.canonicalArgumentsJSONString(parameters)
        )
    }

    private static func extractPlainTextXMLCall(from text: String) -> MLXExtractedToolCall? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(maxSplits: 1, whereSeparator: \.isWhitespace)
        guard let name = parts.first.map(String.init), !name.contains("<") else {
            return nil
        }
        let argumentsText = parts.count > 1 ? String(parts[1]) : ""
        return MLXExtractedToolCall(
            name: name,
            argumentsJSON: Parser.canonicalArgumentsJSONString(Parser.decodedArgumentValue(argumentsText))
        )
    }

    private static func extractGLMXMLCall(from text: String) -> MLXExtractedToolCall? {
        let keyMatches = Parser.captureMatches(
            pattern: #"<arg_key>(.*?)</arg_key>"#,
            in: text,
            captureCount: 1
        )
        let valueMatches = Parser.captureMatches(
            pattern: #"<arg_value>(.*?)</arg_value>"#,
            in: text,
            captureCount: 1
        )
        let keys = keyMatches.compactMap(\.first)
        let values = valueMatches.compactMap(\.first)
        guard let firstKey = keys.first, let range = text.range(of: "<arg_key>") else {
            return nil
        }
        let name = text[..<range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            return nil
        }
        let allKeys = [firstKey] + Array(keys.dropFirst())
        let arguments = Dictionary(uniqueKeysWithValues: zip(allKeys, values).map { pair in
            (pair.0, Parser.decodedArgumentValue(pair.1))
        })
        return MLXExtractedToolCall(name: name, argumentsJSON: Parser.canonicalArgumentsJSONString(arguments))
    }

    static func extractNamespacedToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let pattern = #"<[A-Za-z_][\w.-]*:tool_call>(.*?)</[A-Za-z_][\w.-]*:tool_call>"#
        var calls: [MLXExtractedToolCall] = []
        for block in Parser.captureMatches(pattern: pattern, in: text, captureCount: 1).compactMap(\.first) {
            let invokeCalls = extractInvokeToolCalls(from: block)
            if !invokeCalls.isEmpty {
                calls.append(contentsOf: invokeCalls)
                continue
            }
            calls.append(contentsOf: extractJSONToolCalls(from: block))
        }
        return calls
    }

    static func extractMiniMaxM3ToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let namespace = "]<]minimax[>["
        let blocks = Parser.blocksBetween(
            start: "\(namespace)<tool_call>",
            end: "\(namespace)</tool_call>",
            in: text
        )
        let candidates = blocks.isEmpty && text.contains("\(namespace)<invoke")
            ? [text]
            : blocks
        return candidates.flatMap { block in
            extractMiniMaxM3Invokes(from: block, namespace: namespace)
        }
    }

    private static func extractMiniMaxM3Invokes(
        from text: String,
        namespace: String
    ) -> [MLXExtractedToolCall] {
        let openMarker = "\(namespace)<invoke"
        let closeMarker = "\(namespace)</invoke>"
        var calls: [MLXExtractedToolCall] = []
        var searchStart = text.startIndex

        while let openRange = text.range(of: openMarker, range: searchStart..<text.endIndex),
            let tagEnd = text.range(of: ">", range: openRange.upperBound..<text.endIndex),
            let closeRange = text.range(of: closeMarker, range: tagEnd.upperBound..<text.endIndex) {
            let tag = String(text[openRange.upperBound..<tagEnd.lowerBound])
            if let name = miniMaxM3InvokeName(from: tag) {
                let body = String(text[tagEnd.upperBound..<closeRange.lowerBound])
                calls.append(MLXExtractedToolCall(
                    name: name,
                    argumentsJSON: Parser.canonicalArgumentsJSONString(
                        miniMaxM3DecodedValue(body, namespace: namespace)
                    )
                ))
            }
            searchStart = closeRange.upperBound
        }

        return calls
    }

    private static func miniMaxM3InvokeName(from tag: String) -> String? {
        Parser.captureMatches(pattern: #"name="([^"]+)""#, in: tag, captureCount: 1)
            .first?
            .first
    }

    private static func miniMaxM3DecodedValue(
        _ text: String,
        namespace: String
    ) -> Any {
        let pairs = miniMaxM3ElementPairs(in: text, namespace: namespace)
        guard !pairs.isEmpty else {
            return Parser.decodedArgumentValue(text)
        }
        let containsOnlyArrayItems = pairs.allSatisfy { $0.key == "item" }
        if containsOnlyArrayItems {
            return pairs.map(\.value)
        }
        return Dictionary(pairs.map { ($0.key, $0.value) }) { _, latest in
            latest
        }
    }

    private static func miniMaxM3ElementPairs(
        in text: String,
        namespace: String
    ) -> [(key: String, value: Any)] {
        let openPrefix = "\(namespace)<"
        var pairs: [(key: String, value: Any)] = []
        var searchStart = text.startIndex

        while let openRange = text.range(of: openPrefix, range: searchStart..<text.endIndex),
            let tagEnd = text.range(of: ">", range: openRange.upperBound..<text.endIndex) {
            let tag = String(text[openRange.upperBound..<tagEnd.lowerBound])
            if tag.hasPrefix("/") {
                searchStart = tagEnd.upperBound
                continue
            }
            let closeMarker = "\(namespace)</\(tag)>"
            guard let closeRange = text.range(
                of: closeMarker,
                range: tagEnd.upperBound..<text.endIndex
            ) else {
                break
            }
            let body = String(text[tagEnd.upperBound..<closeRange.lowerBound])
            pairs.append((tag, miniMaxM3DecodedValue(body, namespace: namespace)))
            searchStart = closeRange.upperBound
        }

        return pairs
    }

    private static func extractInvokeToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let matches = Parser.captureMatches(
            pattern: #"<invoke\s+name="([^"]+)">(.*?)</invoke>"#,
            in: text,
            captureCount: 2
        )
        return matches.map { match in
            let arguments = Parser.keyValuePairs(
                pattern: #"<parameter\s+name="([^"]+)">(.*?)</parameter>"#,
                in: match[1]
            )
            return MLXExtractedToolCall(
                name: match[0],
                argumentsJSON: Parser.canonicalArgumentsJSONString(arguments)
            )
        }
    }

    static func extractKimiToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let sections = Parser.blocksBetween(
            start: "<|tool_calls_section_begin|>",
            end: "<|tool_calls_section_end|>",
            in: text
        )
        let candidates = sections.isEmpty && text.contains("<|tool_call_argument_begin|>")
            ? [text]
            : sections
        return candidates.flatMap(extractKimiToolCallsFromSection)
    }

    private static func extractKimiToolCallsFromSection(_ text: String) -> [MLXExtractedToolCall] {
        let blocks = Parser.blocksBetween(
            start: "<|tool_call_begin|>",
            end: "<|tool_call_end|>",
            in: text
        )
        let candidates = blocks.isEmpty ? [text] : blocks
        return candidates.compactMap(extractKimiToolCall)
    }

    private static func extractKimiToolCall(from text: String) -> MLXExtractedToolCall? {
        guard let argumentRange = text.range(of: "<|tool_call_argument_begin|>") else {
            return nil
        }
        let head = String(text[..<argumentRange.lowerBound])
        let matches = Parser.captureMatches(
            pattern: #"^\s*(?:functions\.)?(.+?):\d+\s*$"#,
            in: head,
            captureCount: 1
        )
        guard let name = matches.first?.first?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return nil
        }
        let argumentsText = String(text[argumentRange.upperBound...])
        return MLXExtractedToolCall(
            name: name,
            argumentsJSON: Parser.canonicalArgumentsJSONString(Parser.decodedArgumentValue(argumentsText))
        )
    }

    static func extractLongCatToolCalls(from text: String) -> [MLXExtractedToolCall] {
        Parser.blocksBetween(
            start: "<longcat_tool_call>",
            end: "</longcat_tool_call>",
            in: text
        )
        .flatMap(extractLongCatToolCall)
    }

    private static func extractLongCatToolCall(from text: String) -> [MLXExtractedToolCall] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{") {
            return extractJSONToolCalls(from: trimmed)
        }
        guard let nameRange = trimmed.range(of: "<longcat_arg_key>") else {
            return []
        }
        let name = trimmed[..<nameRange.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            return []
        }
        let pattern = """
        <longcat_arg_key>(.*?)</longcat_arg_key>\\s*\
        <longcat_arg_value>(.*?)</longcat_arg_value>
        """
        let arguments = Parser.keyValuePairs(pattern: pattern, in: trimmed)
        return [
            MLXExtractedToolCall(
                name: String(name),
                argumentsJSON: Parser.canonicalArgumentsJSONString(arguments)
            )
        ]
    }

    static func extractCohereActionToolCalls(from text: String) -> [MLXExtractedToolCall] {
        Parser.blocksBetween(
            start: "<|START_ACTION|>",
            end: "<|END_ACTION|>",
            in: text
        )
        .flatMap { block -> [MLXExtractedToolCall] in
            let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = Parser.parseJSON(trimmed) else {
                return []
            }
            return extractToolCalls(from: value)
        }
    }

    static func extractMistralToolCalls(from text: String) -> [MLXExtractedToolCall] {
        guard let markerRange = text.range(of: "[TOOL_CALLS]") else {
            return []
        }
        let suffix = String(text[markerRange.upperBound...])
        guard let jsonText = Parser.firstJSONValueText(in: suffix, opener: "[", closer: "]")
            ?? Parser.firstJSONValueText(in: suffix, opener: "{", closer: "}"),
            let value = Parser.parseJSON(jsonText) else {
            return extractMistralArgumentCalls(from: suffix)
        }
        return extractToolCalls(from: value)
    }

    private static func extractMistralArgumentCalls(from text: String) -> [MLXExtractedToolCall] {
        let matches = Parser.regexMatches(
            pattern: #"([A-Za-z_][\w.:-]*)\[ARGS\]\s*"#,
            in: text,
            captureCount: 1
        )
        return matches.compactMap { match in
            let suffix = String(text[match.range.upperBound...])
            guard let argumentsText = Parser.firstJSONValueText(in: suffix, opener: "{", closer: "}") else {
                return nil
            }
            return MLXExtractedToolCall(
                name: match.captures[0],
                argumentsJSON: Parser.canonicalArgumentsJSONString(Parser.decodedArgumentValue(argumentsText))
            )
        }
    }
}
