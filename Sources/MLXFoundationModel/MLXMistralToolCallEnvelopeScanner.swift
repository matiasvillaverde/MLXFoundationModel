import Foundation

enum MLXMistralToolCallEnvelopeScanner {
    private struct BalancedScanState {
        var depth = 0
        var isInString = false
        var isEscaped = false
    }

    static func range(in text: String) -> Range<String.Index>? {
        let trimmedStart = text.firstIndex { !$0.isWhitespace } ?? text.endIndex
        guard trimmedStart < text.endIndex else {
            return nil
        }
        if let range = directJSONRange(in: text, at: trimmedStart) {
            return range
        }
        return namedArgumentsRange(in: text, after: trimmedStart)
    }

    private static func directJSONRange(
        in text: String,
        at index: String.Index
    ) -> Range<String.Index>? {
        switch text[index] {
        case "[":
            return balancedRange(in: text, openerIndex: index, opener: "[", closer: "]")

        case "{":
            return balancedRange(in: text, openerIndex: index, opener: "{", closer: "}")

        default:
            return nil
        }
    }

    private static func namedArgumentsRange(
        in text: String,
        after index: String.Index
    ) -> Range<String.Index>? {
        guard let argsRange = text.range(of: "[ARGS]", range: index..<text.endIndex) else {
            return nil
        }
        let jsonStart = text[argsRange.upperBound...].firstIndex { !$0.isWhitespace } ?? text.endIndex
        guard jsonStart < text.endIndex,
            let range = directJSONRange(in: text, at: jsonStart) else {
            return nil
        }
        return text.startIndex..<range.upperBound
    }

    private static func balancedRange(
        in text: String,
        openerIndex: String.Index,
        opener: Character,
        closer: Character
    ) -> Range<String.Index>? {
        var state = BalancedScanState()
        var index = openerIndex
        while index < text.endIndex {
            if scansCompletedJSON(&state, character: text[index], opener: opener, closer: closer) {
                return openerIndex..<text.index(after: index)
            }
            index = text.index(after: index)
        }
        return nil
    }

    private static func scansCompletedJSON(
        _ state: inout BalancedScanState,
        character: Character,
        opener: Character,
        closer: Character
    ) -> Bool {
        if state.isEscaped {
            state.isEscaped = false
            return false
        }
        if character == "\\" {
            state.isEscaped = state.isInString
            return false
        }
        if character == "\"" {
            state.isInString.toggle()
            return false
        }
        return scansCompletedStructuralJSON(&state, character: character, opener: opener, closer: closer)
    }

    private static func scansCompletedStructuralJSON(
        _ state: inout BalancedScanState,
        character: Character,
        opener: Character,
        closer: Character
    ) -> Bool {
        guard !state.isInString else {
            return false
        }
        if character == opener {
            state.depth += 1
        } else if character == closer {
            state.depth -= 1
        }
        return state.depth == 0
    }
}
