import Foundation

extension MLXBalancedPrefixScanner {
    func matchingCloser(for character: Character) -> Character? {
        switch character {
        case "{":
            return "}"

        case "[":
            return "]"

        case "(":
            return ")"

        default:
            return nil
        }
    }
}
