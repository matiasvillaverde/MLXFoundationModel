struct MLXToolCallSplitState {
    private var depth = 0
    private var inString = false
    private var stringDelimiter: Character?
    private var isEscaped = false

    var isTopLevel: Bool {
        !inString && depth == 0
    }

    mutating func update(with character: Character) {
        if updateString(character) {
            return
        }
        switch character {
        case "{", "[", "(":
            depth += 1

        case "}", "]", ")":
            depth = max(depth - 1, 0)

        default:
            break
        }
    }

    private mutating func updateString(_ character: Character) -> Bool {
        if isEscaped {
            isEscaped = false
            return true
        }
        if character == "\\" {
            isEscaped = true
            return true
        }
        if inString, character == stringDelimiter {
            inString = false
            stringDelimiter = nil
            return true
        }
        if !inString, character == "\"" || character == "'" {
            inString = true
            stringDelimiter = character
            return true
        }
        return inString
    }
}
