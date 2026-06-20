import Foundation

extension MLXToolCallParsingSupport {
    static func balancedPrefix(
        in text: String,
        opener: Character,
        closer: Character
    ) -> String? {
        balancedPrefix(
            in: text,
            from: text.startIndex,
            opener: opener,
            closer: closer
        )
    }

    static func balancedPrefix(
        in text: String,
        from startIndex: String.Index,
        opener: Character,
        closer: Character
    ) -> String? {
        let scanner = MLXBalancedPrefixScanner(text: text)
        return scanner.scan(from: startIndex, opener: opener, closer: closer)
    }
}
