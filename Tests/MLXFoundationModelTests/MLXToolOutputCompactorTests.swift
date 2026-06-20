@testable import MLXFoundationModel
import Testing

@Suite("MLX tool output compactor")
struct MLXToolOutputCompactorTests {
    @Test("leaves small tool outputs unchanged")
    func leavesSmallToolOutputsUnchanged() {
        let text = "short result"

        #expect(MLXToolOutputCompactor.compact(text, limit: 128) == text)
    }

    @Test("keeps compacted tool output within limit")
    func keepsCompactedToolOutputWithinLimit() {
        let text = "prefix-" + String(repeating: "a", count: 160) + "-suffix"

        let compacted = MLXToolOutputCompactor.compact(text, limit: 96)

        #expect(compacted.count <= 96)
        #expect(compacted.hasPrefix("prefix-"))
        #expect(compacted.hasSuffix("-suffix"))
        #expect(compacted.contains("MLXFoundationModel truncated"))
        #expect(!compacted.contains(String(repeating: "a", count: 120)))
    }

    @Test("handles tiny limits without exceeding them")
    func handlesTinyLimitsWithoutExceedingThem() {
        let compacted = MLXToolOutputCompactor.compact("abcdef", limit: 4)

        #expect(compacted.count == 4)
    }

    @Test("handles non-positive limits as empty output")
    func handlesNonPositiveLimitsAsEmptyOutput() {
        #expect(MLXToolOutputCompactor.compact("abcdef", limit: 0).isEmpty)
        #expect(MLXToolOutputCompactor.compact("abcdef", limit: -4).isEmpty)
    }

    @Test("reports actual omitted character count")
    func reportsActualOmittedCharacterCount() throws {
        let text = "prefix-" + String(repeating: "a", count: 160) + "-suffix"

        let compacted = MLXToolOutputCompactor.compact(text, limit: 96)
        let markerRange = try #require(Self.truncationMarkerRange(in: compacted))
        let omitted = try #require(Self.omittedCharacterCount(in: compacted))
        let keptCount = compacted.count - compacted[markerRange].count

        #expect(omitted == text.count - keptCount)
        #expect(compacted.count <= 96)
    }

    private static func truncationMarkerRange(in text: String) -> Range<String.Index>? {
        guard
            let start = text.range(of: "\n\n[MLXFoundationModel truncated "),
            let end = text.range(
                of: " characters from this tool output]\n\n",
                range: start.upperBound..<text.endIndex
            )
        else {
            return nil
        }
        return start.lowerBound..<end.upperBound
    }

    private static func omittedCharacterCount(in text: String) -> Int? {
        guard
            let prefix = text.range(of: "[MLXFoundationModel truncated "),
            let suffix = text.range(
                of: " characters from this tool output]",
                range: prefix.upperBound..<text.endIndex
            )
        else {
            return nil
        }
        return Int(text[prefix.upperBound..<suffix.lowerBound])
    }
}
