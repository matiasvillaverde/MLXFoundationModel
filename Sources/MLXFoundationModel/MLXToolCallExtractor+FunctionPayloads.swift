import Foundation

extension MLXToolCallExtractor {
    static func extractHermesToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let pattern = #"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>"#
        var calls: [MLXExtractedToolCall] = []
        for block in Parser.captureMatches(pattern: pattern, in: text, captureCount: 1).compactMap(\.first) {
            let jsonCalls = extractJSONToolCalls(from: block)
            if !jsonCalls.isEmpty {
                calls.append(contentsOf: jsonCalls)
                continue
            }
            calls.append(contentsOf: extractBracketPayloadToolCalls(from: block))
        }
        return calls
    }

    static func extractBracketToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let patternWithArguments = #"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)\((.*?)\)\]"#
        let argumentMatches = Parser.regexMatches(
            pattern: patternWithArguments,
            in: text,
            captureCount: 2
        )
        var calls = argumentMatches.map { match in
            MLXExtractedToolCall(
                name: match.captures[0],
                argumentsJSON: Parser.canonicalArgumentsJSONString(
                    Parser.parsedCallArguments(match.captures[1])
                )
            )
        }

        let patternWithoutArguments = #"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)\]"#
        for match in Parser.regexMatches(pattern: patternWithoutArguments, in: text, captureCount: 1) {
            guard !argumentMatches.contains(where: { $0.range.overlaps(match.range) }) else {
                continue
            }
            calls.append(MLXExtractedToolCall(
                name: match.captures[0],
                argumentsJSON: Parser.canonicalArgumentsJSONString([:])
            ))
        }
        return calls
    }

    static func extractGemmaToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let markedBlocks = Parser.blocksBetween(
            start: "<|tool_call>",
            end: "<tool_call|>",
            in: text
        )
        let legacyBlocks = Parser.blocksBetween(
            start: "<start_function_call>",
            end: "<end_function_call>",
            in: text
        )
        guard !markedBlocks.isEmpty || !legacyBlocks.isEmpty || text.contains("call:") else {
            return []
        }
        let blocks = markedBlocks + legacyBlocks
        let payloads = blocks.isEmpty ? [text] : blocks
        var calls: [MLXExtractedToolCall] = []
        for block in payloads {
            calls.append(contentsOf: extractFunctionCallPayloads(block))
        }
        return calls
    }

    private static func extractBracketPayloadToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let payload = trimmed.hasPrefix("[") && trimmed.hasSuffix("]")
            ? String(trimmed.dropFirst().dropLast())
            : trimmed
        return extractFunctionCallPayloads(payload)
    }

    private static func extractFunctionCallPayloads(_ payload: String) -> [MLXExtractedToolCall] {
        guard payload.utf8.count <= 262_144 else {
            return []
        }

        var calls: [MLXExtractedToolCall] = []
        var consumedUntil = payload.startIndex
        var searchStart = payload.startIndex

        while let head = nextFunctionCallHead(in: payload, from: searchStart) {
            searchStart = payload.index(after: head.nameStart)
            guard head.openerIndex >= consumedUntil else {
                continue
            }
            guard let call = extractedCall(from: head, in: payload) else {
                continue
            }
            calls.append(call.value)
            consumedUntil = call.endIndex
            searchStart = consumedUntil
        }
        return calls
    }

    private struct ParsedFunctionCall {
        let value: MLXExtractedToolCall
        let endIndex: String.Index
    }

    private struct FunctionCallHead {
        let name: String
        let nameStart: String.Index
        let opener: Character
        let openerIndex: String.Index
    }

    private static func extractedCall(
        from head: FunctionCallHead,
        in payload: String
    ) -> ParsedFunctionCall? {
        let closer: Character = head.opener == "{" ? "}" : ")"
        guard let argumentsText = Parser.balancedPrefix(
            in: payload,
            from: head.openerIndex,
            opener: head.opener,
            closer: closer
        ) else {
            return nil
        }

        let arguments = head.opener == "{"
            ? Parser.parsedObjectLiteral(argumentsText)
            : Parser.parsedCallArguments(argumentsText)
        return ParsedFunctionCall(
            value: MLXExtractedToolCall(
                name: head.name,
                argumentsJSON: Parser.canonicalArgumentsJSONString(arguments)
            ),
            endIndex: payload.index(head.openerIndex, offsetBy: argumentsText.count)
        )
    }

    private static func nextFunctionCallHead(
        in payload: String,
        from startIndex: String.Index
    ) -> FunctionCallHead? {
        var index = startIndex
        while index < payload.endIndex {
            if let head = functionCallHead(in: payload, at: index) {
                return head
            }
            index = payload.index(after: index)
        }
        return nil
    }

    private static func functionCallHead(
        in payload: String,
        at index: String.Index
    ) -> FunctionCallHead? {
        let nameStart = functionNameStart(in: payload, at: index)
        guard let nameEnd = functionNameEnd(in: payload, from: nameStart),
            nameEnd > nameStart else {
            return nil
        }

        var cursor = nameEnd
        while cursor < payload.endIndex, payload[cursor].isWhitespace {
            cursor = payload.index(after: cursor)
        }
        guard cursor < payload.endIndex,
            payload[cursor] == "{" || payload[cursor] == "(" else {
            return nil
        }

        return FunctionCallHead(
            name: String(payload[nameStart..<nameEnd]),
            nameStart: nameStart,
            opener: payload[cursor],
            openerIndex: cursor
        )
    }

    private static func functionNameStart(
        in payload: String,
        at index: String.Index
    ) -> String.Index {
        if payload[index...].hasPrefix("call:") {
            return payload.index(index, offsetBy: "call:".count)
        }
        if payload[index...].hasPrefix("call") {
            let start = payload.index(index, offsetBy: "call".count)
            if start < payload.endIndex, isFunctionNameStart(payload[start]) {
                return start
            }
        }
        if payload[index] == ":" {
            return payload.index(after: index)
        }
        return index
    }

    private static func functionNameEnd(
        in payload: String,
        from startIndex: String.Index
    ) -> String.Index? {
        guard startIndex < payload.endIndex,
            isFunctionNameStart(payload[startIndex]) else {
            return nil
        }

        var index = payload.index(after: startIndex)
        while index < payload.endIndex {
            let character = payload[index]
            if isFunctionNameBody(character) {
                index = payload.index(after: index)
                continue
            }
            if character == ":" {
                let nextIndex = payload.index(after: index)
                guard nextIndex < payload.endIndex,
                    isFunctionNameStart(payload[nextIndex]) else {
                    break
                }
                index = payload.index(after: nextIndex)
                continue
            }
            break
        }
        return index
    }

    private static func isFunctionNameStart(_ character: Character) -> Bool {
        character == "_" || character.isASCII && character.isLetter
    }

    private static func isFunctionNameBody(_ character: Character) -> Bool {
        character == "_"
            || character == "."
            || character == "-"
            || character.isASCII && (character.isLetter || character.isNumber)
    }
}
