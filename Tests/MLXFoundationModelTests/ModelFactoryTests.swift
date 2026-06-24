import Foundation
import Hub
@testable import MLXLocalModels
import Testing
import Tokenizers

// swiftlint:disable discouraged_optional_collection file_types_order function_parameter_count
// swiftlint:disable identifier_name one_declaration_per_file unneeded_escaping
@Suite("Model factory")
struct ModelFactoryTests {
    @Test("plain prompts use chat templates when available")
    func plainPromptsUseChatTemplatesWhenAvailable() throws {
        let tokenizer = FactoryTokenizer(hasChatTemplate: true)

        let input = try PromptTokenizer.tokenize(prompt: "hello", tokenizer: tokenizer)

        #expect(input.text.tokens.asArray(Int.self) == [1_005])
    }

    @Test("preformatted prompts bypass chat templates")
    func preformattedPromptsBypassChatTemplates() throws {
        let tokenizer = FactoryTokenizer(hasChatTemplate: true)

        let input = try PromptTokenizer.tokenize(prompt: "<|im_start|>hello", tokenizer: tokenizer)

        #expect(input.text.tokens.asArray(Int.self) == [17])
    }

    @Test("cache identified prompts are encoded verbatim")
    func cacheIdentifiedPromptsAreEncodedVerbatim() throws {
        let tokenizer = FactoryTokenizer(hasChatTemplate: true)
        let renderedInput = LLMInput(
            context: "hello",
            promptCacheIdentity: PromptCacheIdentity(stableFingerprint: "factory-test")
        )

        let input = try PromptTokenizer.tokenize(input: renderedInput, tokenizer: tokenizer)

        #expect(input.text.tokens.asArray(Int.self) == [5])
    }

    @Test("non-chat tokenizers encode plain prompts")
    func nonChatTokenizersEncodePlainPrompts() throws {
        let tokenizer = FactoryTokenizer(hasChatTemplate: false)

        let input = try PromptTokenizer.tokenize(prompt: "hello", tokenizer: tokenizer)

        #expect(input.text.tokens.asArray(Int.self) == [5])
    }

    @Test("dispatcher tries later factories after earlier failures")
    func dispatcherTriesLaterFactoriesAfterEarlierFailures() async throws {
        let dispatcher = ModelLoadDispatcher(
            registry: ModelFactoryRegistry(trampolines: [
                { StubModelFactory(.failure(.first)) },
                { nil },
                { StubModelFactory(.success("loaded/model")) }
            ])
        )

        let context = try await dispatcher.loadContext(
            configuration: ModelConfiguration(id: "requested/model")
        )

        #expect(context.configuration.name == "loaded/model")
    }

    @Test("dispatcher throws the last factory error")
    func dispatcherThrowsLastFactoryError() async throws {
        let dispatcher = ModelLoadDispatcher(registry: ModelFactoryRegistry(trampolines: [
            { StubModelFactory(.failure(.first)) },
            { StubModelFactory(.failure(.second)) }
        ]))

        do {
            _ = try await dispatcher.loadContext(configuration: ModelConfiguration(id: "requested/model"))
            Issue.record("Expected final factory error")
        } catch let error as StubFactoryError {
            #expect(error == .second)
        }
    }

    @Test("dispatcher reports missing factories")
    func dispatcherReportsMissingFactories() async throws {
        let dispatcher = ModelLoadDispatcher(registry: ModelFactoryRegistry(trampolines: []))

        do {
            _ = try await dispatcher.loadContext(configuration: ModelConfiguration(id: "requested/model"))
            Issue.record("Expected no factory error")
        } catch ModelFactoryError.noModelFactoryAvailable {
            return
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}

private struct FactoryTokenizer: Tokenizer {
    let hasChatTemplate: Bool

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<eos>" }
    var eosTokenId: Int? { 2 }
    var unknownToken: String? { "<unk>" }
    var unknownTokenId: Int? { 0 }

    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map(String.init)
    }

    func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [text.count]
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? eosTokenId : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == eosTokenId ? eosToken : nil
    }

    func applyChatTemplate(messages: [Message]) throws -> [Int] {
        let characterCount = messages
            .compactMap { $0["content"] as? String }
            .joined(separator: " ")
            .count
        return [1_000 + characterCount]
    }

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Message],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }
}

private enum StubFactoryResult {
    case failure(StubFactoryError)
    case success(String)
}

private enum StubFactoryError: Error, Equatable {
    case first
    case second
}

private final class StubModelFactory: ModelFactory, @unchecked Sendable {
    let modelRegistry = AbstractModelRegistry()
    private let result: StubFactoryResult

    deinit {
        // Required by the strict test lint profile.
    }

    init(_ result: StubFactoryResult) {
        self.result = result
    }

    func _load(
        hub: HubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext {
        switch result {
        case .failure(let error):
            throw error

        case .success(let modelID):
            return ModelContext(
                configuration: ModelConfiguration(id: modelID),
                model: MLXEchoBatchLanguageModel(),
                tokenizer: FactoryTokenizer(hasChatTemplate: false)
            )
        }
    }

    func _loadContainer(
        hub: HubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContainer {
        ModelContainer(
            context: try await _load(
                hub: hub,
                configuration: configuration,
                progressHandler: progressHandler
            )
        )
    }
}
// swiftlint:enable discouraged_optional_collection file_types_order function_parameter_count
// swiftlint:enable identifier_name one_declaration_per_file unneeded_escaping
