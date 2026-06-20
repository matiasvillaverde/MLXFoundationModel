import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX prompt cache reuse alignment")
struct MLXPromptCacheReuseAlignmentTests {
    @Test("exact alignment reuses the longest safe partial prefix")
    func exactAlignmentReusesLongestSafePartialPrefix() throws {
        var entries = Self.entries()

        let plan = Self.plan(alignment: .exact, entries: &entries)

        #expect(plan.reusedTokenCount == 9)
        #expect(plan.input.text.tokens.asArray(Int.self) == [99])
        #expect(plan.cache?.first?.offset == 9)
    }

    @Test("prefill-step alignment rounds partial prefix reuse down")
    func prefillStepAlignmentRoundsPartialPrefixReuseDown() throws {
        var entries = Self.entries()

        let plan = Self.plan(alignment: .prefillStep, entries: &entries)

        #expect(plan.reusedTokenCount == 8)
        #expect(plan.input.text.tokens.asArray(Int.self) == [8, 99])
        #expect(plan.cache?.first?.offset == 8)
    }

    @Test("prefill-step alignment preserves exact hit generation kickoff")
    func prefillStepAlignmentPreservesExactHitGenerationKickoff() throws {
        var entries = [
            PromptCacheEntry(
                tokens: Array(0 ..< 10),
                cache: [PromptCacheTestCache(offset: 10)],
                signature: PromptCacheSignature(
                    parameters: GenerateParameters(prefillStepSize: 4)
                ),
                byteCount: 10
            )
        ]

        let parameters = GenerateParameters(
            prefillStepSize: 4,
            promptCacheReuseAlignment: .prefillStep
        )
        let plan = PromptCachePlanner.plan(
            fullInput: LMInput(tokens: MLXArray(Array(0 ..< 10))),
            tokenIds: Array(0 ..< 10),
            parameters: parameters,
            existingEntries: &entries,
            reuseEnabled: true
        )

        #expect(plan.reusedTokenCount == 9)
        #expect(plan.input.text.tokens.asArray(Int.self) == [9])
        #expect(plan.cache?.first?.offset == 9)
    }

    private static func plan(
        alignment: PromptCacheReuseAlignment,
        entries: inout [PromptCacheEntry]
    ) -> PromptCachePlan {
        let parameters = GenerateParameters(
            prefillStepSize: 4,
            promptCacheReuseAlignment: alignment
        )
        return PromptCachePlanner.plan(
            fullInput: LMInput(tokens: MLXArray(Self.requestTokens)),
            tokenIds: Self.requestTokens,
            parameters: parameters,
            existingEntries: &entries,
            reuseEnabled: true
        )
    }

    private static func entries() -> [PromptCacheEntry] {
        [
            PromptCacheEntry(
                tokens: Array(0 ..< 10),
                cache: [PromptCacheTestCache(offset: 10)],
                signature: PromptCacheSignature(
                    parameters: GenerateParameters(prefillStepSize: 4)
                ),
                byteCount: 10
            )
        ]
    }

    private static let requestTokens = Array(0 ..< 9) + [99]
}
