import Foundation

extension MLXToolCallExtractor {
    static func extractDeepSeekDSMLToolCalls(from text: String) -> [MLXExtractedToolCall] {
        let blocks = Parser.blocksBetween(
            start: "<｜DSML｜tool_calls>",
            end: "</｜DSML｜tool_calls>",
            in: text
        )
        let candidates = blocks.isEmpty && text.contains("<｜DSML｜invoke")
            ? [text]
            : blocks
        return candidates.flatMap(extractDeepSeekInvokes)
    }

    private static func extractDeepSeekInvokes(from text: String) -> [MLXExtractedToolCall] {
        Parser.captureMatches(
            pattern: #"<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>"#,
            in: text,
            captureCount: 2
        )
        .map { match in
            MLXExtractedToolCall(
                name: match[0],
                argumentsJSON: Parser.canonicalArgumentsJSONString(
                    deepSeekArguments(from: match[1])
                )
            )
        }
    }

    private static func deepSeekArguments(from text: String) -> JSONObject {
        let matches = Parser.captureMatches(
            pattern: #"<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>"#,
            in: text,
            captureCount: 3
        )
        return Dictionary(uniqueKeysWithValues: matches.map { match in
            (
                match[0],
                deepSeekDecodedValue(match[2], isString: match[1] == "true")
            )
        })
    }

    private static func deepSeekDecodedValue(
        _ value: String,
        isString: Bool
    ) -> Any {
        let normalized = value.trimmingOneBoundaryNewline()
        return isString ? normalized : Parser.decodedArgumentValue(normalized)
    }
}
