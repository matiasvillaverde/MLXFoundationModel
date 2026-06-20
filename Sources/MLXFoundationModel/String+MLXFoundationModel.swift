import Foundation

extension String {
    func droppingJSONValueEnvelope() -> String {
        let prefix = #"{"value":"#
        guard hasPrefix(prefix), hasSuffix("}") else {
            return self
        }
        return String(dropFirst(prefix.count).dropLast())
    }

    func suffixString(retainingAtMost count: Int) -> String {
        guard count > 0, self.count > count else {
            return self
        }
        let start = index(endIndex, offsetBy: -count)
        return String(self[start...])
    }

    func trimmingOneBoundaryNewline() -> String {
        var text = self
        if text.first == "\n" {
            text.removeFirst()
        }
        if text.last == "\n" {
            text.removeLast()
        }
        return text
    }
}
