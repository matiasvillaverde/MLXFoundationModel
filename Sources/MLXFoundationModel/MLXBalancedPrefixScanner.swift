import Foundation

struct MLXBalancedPrefixScanner {
    private static let maximumLength = 262_144
    private static let maximumDepth = 64

    let text: String

    func scan(
        from startIndex: String.Index,
        opener: Character,
        closer: Character
    ) -> String? {
        guard startIndex < text.endIndex, text[startIndex] == opener else {
            return nil
        }

        var expectedClosers: [Character] = []
        var index = startIndex
        var scannedLength = 0
        while index < text.endIndex {
            guard consumeNext(
                index: &index,
                scannedLength: &scannedLength,
                expectedClosers: &expectedClosers
            ) else {
                return nil
            }
            if expectedClosers.isEmpty {
                return String(text[startIndex...index])
            }
            index = text.index(after: index)
        }
        return nil
    }

    private func consumeNext(
        index: inout String.Index,
        scannedLength: inout Int,
        expectedClosers: inout [Character]
    ) -> Bool {
        scannedLength += 1
        guard scannedLength <= Self.maximumLength else {
            return false
        }
        if let skipped = skippedStringEnd(from: index) {
            scannedLength += text.distance(from: index, to: skipped) - 1
            index = text.index(before: skipped)
            return scannedLength <= Self.maximumLength
        }
        return consumeStructure(at: index, expectedClosers: &expectedClosers)
    }

    private func consumeStructure(
        at index: String.Index,
        expectedClosers: inout [Character]
    ) -> Bool {
        let character = text[index]
        if let closer = matchingCloser(for: character) {
            expectedClosers.append(closer)
            return expectedClosers.count <= Self.maximumDepth
        }
        if character == expectedClosers.last {
            expectedClosers.removeLast()
        }
        return true
    }
}
