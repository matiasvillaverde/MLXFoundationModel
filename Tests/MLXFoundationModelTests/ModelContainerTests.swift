@testable import MLXLocalModels
import Testing

@Suite("Model container")
struct ModelContainerTests {
    @Test("exposes configuration and updates context inside the actor")
    func exposesConfigurationAndUpdatesContext() async {
        let container = ModelContainer(context: Self.context(id: "test/before"))

        #expect(await container.configuration.name == "test/before")

        await container.update { context in
            context.configuration = ModelConfiguration(id: "test/after", eosTokenIds: [42])
        }

        #expect(await container.configuration.name == "test/after")
        let tokenIDs = await container.perform { context in
            context.configuration.eosTokenIds
        }
        #expect(tokenIDs == [42])
    }

    @Test("forwards context and caller values through perform")
    func forwardsContextAndValuesThroughPerform() async throws {
        let container = ModelContainer(context: Self.context(id: "test/perform"))

        let rendered = try await container.perform(values: "hello world") { context, prompt in
            try context.tokenize(prompt: prompt).text.tokens.asArray(Int.self)
        }

        #expect(rendered == [10, 11])
    }

    @Test("keeps legacy model-tokenizer perform overloads compatible")
    @available(*, deprecated, message: "Exercises deprecated ModelContainer compatibility overloads.")
    func keepsLegacyPerformOverloadsCompatible() async {
        let container = ModelContainer(context: Self.context(id: "test/legacy"))

        let encoded = await container.perform { _, tokenizer in
            tokenizer.encode(text: "hello")
        }
        let decoded = await container.perform(values: [10, 11]) { _, tokenizer, tokenIDs in
            tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: false)
        }

        #expect(encoded == [10])
        #expect(decoded == "helloworld")
    }

    @Test("persists prompt cache mutations and supports snapshot replacement")
    func persistsPromptCacheMutationsAndReplacement() async {
        let container = ModelContainer(context: Self.context(id: "test/cache"))
        let firstEntry = Self.entry(tokens: [1, 2, 3])
        let replacementEntry = Self.entry(tokens: [9])

        let count = await container.performWithPromptCache { _, entries in
            entries.append(firstEntry)
            return entries.count
        }
        #expect(count == 1)
        #expect(await container.promptCacheEntriesSnapshot().map(\.tokens) == [[1, 2, 3]])

        await container.replacePromptCacheEntries([replacementEntry])
        #expect(await container.promptCacheEntriesSnapshot().map(\.tokens) == [[9]])

        await container.replacePromptCacheEntries([])
        #expect(await container.promptCacheEntriesSnapshot().isEmpty)

        await container.replacePromptCacheEntries([firstEntry])
        await container.clearPromptCache()
        #expect(await container.promptCacheEntriesSnapshot().isEmpty)
    }

    private static func context(id: String) -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(id: id, eosTokenIds: [99]),
            model: MLXEchoBatchLanguageModel(),
            tokenizer: PreparedGenerationTokenizer()
        )
    }

    private static func entry(tokens: [Int]) -> PromptCacheEntry {
        PromptCacheEntry(
            tokens: tokens,
            cache: [],
            signature: PromptCacheSignature(parameters: GenerateParameters()),
            byteCount: 0
        )
    }
}
