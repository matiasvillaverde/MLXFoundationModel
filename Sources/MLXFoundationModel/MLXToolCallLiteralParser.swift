import Foundation

enum MLXToolCallLiteralParser {
    static let maximumLength = 262_144
    static let maximumDepth = 64

    static func parseArguments(_ text: String) -> Any? {
        var parser = MLXToolCallLiteralParserState(text)
        parser.skipWhitespace()

        let result: Any?
        switch parser.peek {
        case "{":
            result = parser.parseObject(open: "{", close: "}", depth: 0)

        case "(":
            result = parser.parseObject(open: "(", close: ")", depth: 0)

        default:
            result = parser.parseEntries(close: nil, depth: 0)
        }

        parser.skipWhitespace()
        return parser.isAtEnd ? result : nil
    }

    static func parseValue(_ text: String) -> Any? {
        var parser = MLXToolCallLiteralParserState(text)
        parser.skipWhitespace()
        guard let value = parser.parseValue(depth: 0) else {
            return nil
        }
        parser.skipWhitespace()
        return parser.isAtEnd ? value : nil
    }
}
