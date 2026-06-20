@testable import MLXLocalModels
import Testing

@Suite("MLX prompt cache block index")
struct MLXPromptCacheBlockIndexTests {
    @Test("hash chains disambiguate identical later blocks")
    func hashChainsDisambiguateIdenticalLaterBlocks() throws {
        let signature = Self.signature()
        let entries = [
            Self.entry(tokens: [1, 2, 3, 4, 5, 6, 7, 8], signature: signature),
            Self.entry(tokens: [9, 9, 9, 9, 5, 6, 7, 8], signature: signature)
        ]
        let index = PromptCacheBlockIndex(
            entries: entries,
            signature: signature,
            requiresDraftCache: false,
            blockSize: 4
        )

        let lookup = try #require(index.lookup(tokenIds: [1, 2, 3, 4, 5, 6, 7, 8]))

        #expect(lookup.matchedBlockCount == 2)
        #expect(lookup.candidateIndexes == [0])
    }

    @Test("planner uses block index for long compatible prompts")
    func plannerUsesBlockIndexForLongCompatiblePrompts() async throws {
        let tokenIds = Array(0 ..< 301)
        let entries = [
            Self.entry(tokens: Array(0 ..< 300), signature: Self.signature())
        ]

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            PromptCachePlanner.bestCandidate(
                tokenIds: tokenIds,
                parameters: GenerateParameters(),
                existingEntries: entries,
                requiresDraftCache: false
            )
        }

        #expect(recorded.result?.reusableTokenCount == 300)
        #expect(Self.lookupSnapshots(from: recorded.events).contains { snapshot in
            snapshot.strategy == .blockIndex && snapshot.reusedTokenCount == 300
        })
    }

    @Test("planner falls back to linear lookup for short prompts")
    func plannerFallsBackToLinearLookupForShortPrompts() async throws {
        let tokenIds = [1, 2, 3]
        let entries = [
            Self.entry(tokens: [1, 2], signature: Self.signature())
        ]

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            PromptCachePlanner.bestCandidate(
                tokenIds: tokenIds,
                parameters: GenerateParameters(),
                existingEntries: entries,
                requiresDraftCache: false
            )
        }

        #expect(recorded.result?.reusableTokenCount == 2)
        #expect(Self.lookupSnapshots(from: recorded.events).contains { snapshot in
            snapshot.strategy == .linear && snapshot.reusedTokenCount == 2
        })
    }

    private static func lookupSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPromptCacheLookupSnapshot] {
        events.compactMap { event in
            guard case .promptCacheLookup(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func entry(
        tokens: [Int],
        signature: PromptCacheSignature
    ) -> PromptCacheEntry {
        PromptCacheEntry(
            tokens: tokens,
            cache: [],
            signature: signature,
            byteCount: 0
        )
    }

    private static func signature() -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters())
    }
}
