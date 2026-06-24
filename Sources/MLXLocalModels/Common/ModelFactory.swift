import Foundation
import Hub
import MLX
import Tokenizers

internal enum ModelFactoryError: LocalizedError {
    case unsupportedModelType(String)
    case unsupportedProcessorType(String)
    case configurationDecodingError(String, String, DecodingError)
    case noModelFactoryAvailable

    internal var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let modelType):
            "Unsupported model type: \(modelType)"
        case .unsupportedProcessorType(let processorType):
            "Unsupported processor type: \(processorType)"
        case .configurationDecodingError(let fileName, let modelName, let decodingError):
            "Failed to parse \(fileName) for model '\(modelName)': "
                + DecodingErrorFormatter.message(for: decodingError)
        case .noModelFactoryAvailable:
            "No model factory available via ModelFactoryRegistry"
        }
    }
}

private enum DecodingErrorFormatter {
    static func message(for error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, let context):
            "Missing field '\(path(context.codingPath + [key]))'"
        case .typeMismatch(_, let context):
            "Type mismatch at '\(path(context.codingPath))'"
        case .valueNotFound(_, let context):
            "Missing value at '\(path(context.codingPath))'"
        case .dataCorrupted(let context):
            context.codingPath.isEmpty ? "Invalid JSON" : "Invalid data at '\(path(context.codingPath))'"
        @unknown default:
            error.localizedDescription
        }
    }

    private static func path(_ codingPath: [CodingKey]) -> String {
        codingPath.map(\.stringValue).joined(separator: ".")
    }
}

internal struct ModelContext: @unchecked Sendable {
    internal var configuration: ModelConfiguration
    internal var model: any LanguageModel
    internal var tokenizer: any Tokenizer
    internal var grammarCompiler: GrammarConstraintCompiler?
    internal var grammarCompilerError: Error?

    internal init(
        configuration: ModelConfiguration,
        model: any LanguageModel,
        tokenizer: any Tokenizer,
        grammarCompiler: GrammarConstraintCompiler? = nil,
        grammarCompilerError: Error? = nil
    ) {
        self.configuration = configuration
        self.model = model
        self.tokenizer = tokenizer
        self.grammarCompiler = grammarCompiler
        self.grammarCompilerError = grammarCompilerError
    }

    internal func tokenize(input: LLMInput) throws -> LMInput {
        try PromptTokenizer.tokenize(input: input, tokenizer: tokenizer)
    }

    internal func tokenize(prompt: String) throws -> LMInput {
        try PromptTokenizer.tokenize(prompt: prompt, tokenizer: tokenizer)
    }
}

internal enum PromptTokenizer {
    internal static func tokenize(
        input: LLMInput,
        tokenizer: any Tokenizer
    ) throws -> LMInput {
        if input.promptMetadata != nil || input.promptCacheIdentity != nil {
            return encode(input.context, with: tokenizer)
        }
        return try tokenize(prompt: input.context, tokenizer: tokenizer)
    }

    internal static func tokenize(
        prompt: String,
        tokenizer: any Tokenizer
    ) throws -> LMInput {
        guard shouldApplyChatTemplate(to: prompt, tokenizer: tokenizer) else {
            return encode(prompt, with: tokenizer)
        }

        let messages: [Message] = [
            ["role": "user", "content": prompt]
        ]
        return LMInput(tokens: MLXArray(try tokenizer.applyChatTemplate(messages: messages)))
    }

    private static func shouldApplyChatTemplate(
        to prompt: String,
        tokenizer: any Tokenizer
    ) -> Bool {
        tokenizer.hasChatTemplate && !PromptFormatDetector.isPreformatted(prompt)
    }

    private static func encode(
        _ prompt: String,
        with tokenizer: any Tokenizer
    ) -> LMInput {
        LMInput(tokens: MLXArray(tokenizer.encode(text: prompt)))
    }
}

internal protocol ModelFactory: Sendable {
    var modelRegistry: AbstractModelRegistry { get }

    func _load(
        hub: HubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext

    func _loadContainer(
        hub: HubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContainer
}

extension ModelFactory {
    internal func configuration(id: String) -> ModelConfiguration {
        modelRegistry.configuration(id: id)
    }

    internal func contains(id: String) -> Bool {
        modelRegistry.contains(id: id)
    }

    internal func load(
        hub: HubApi = defaultHubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ModelContext {
        try await _load(hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    internal func loadContainer(
        hub: HubApi = defaultHubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        try await _loadContainer(
            hub: hub,
            configuration: configuration,
            progressHandler: progressHandler
        )
    }

    internal func _loadContainer(
        hub: HubApi = defaultHubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
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

internal nonisolated(unsafe) var defaultHubApi: HubApi = {
    HubApi(downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first)
}()

internal struct ModelLoadDispatcher: Sendable {
    private let registry: ModelFactoryRegistry

    internal init(registry: ModelFactoryRegistry = .shared) {
        self.registry = registry
    }

    internal func loadContext(
        hub: HubApi = defaultHubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ModelContext {
        try await resolve { factory in
            try await factory.load(
                hub: hub,
                configuration: configuration,
                progressHandler: progressHandler
            )
        }
    }

    internal func loadContainer(
        hub: HubApi = defaultHubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ModelContainer {
        try await resolve { factory in
            try await factory.loadContainer(
                hub: hub,
                configuration: configuration,
                progressHandler: progressHandler
            )
        }
    }

    private func resolve<R>(
        _ operation: (ModelFactory) async throws -> sending R
    ) async throws -> sending R {
        var lastError: Error?
        for factory in registry.modelFactories() {
            do {
                return try await operation(factory)
            } catch {
                lastError = error
            }
        }
        throw lastError ?? ModelFactoryError.noModelFactoryAvailable
    }
}

internal func loadModel(
    hub: HubApi = defaultHubApi,
    configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await ModelLoadDispatcher().loadContext(
        hub: hub,
        configuration: configuration,
        progressHandler: progressHandler
    )
}

internal func loadModelContainer(
    hub: HubApi = defaultHubApi,
    configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await ModelLoadDispatcher().loadContainer(
        hub: hub,
        configuration: configuration,
        progressHandler: progressHandler
    )
}

internal func loadModel(
    hub: HubApi = defaultHubApi,
    id: String,
    revision: String = "main",
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await loadModel(
        hub: hub,
        configuration: ModelConfiguration(id: id, revision: revision),
        progressHandler: progressHandler
    )
}

internal func loadModelContainer(
    hub: HubApi = defaultHubApi,
    id: String,
    revision: String = "main",
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await loadModelContainer(
        hub: hub,
        configuration: ModelConfiguration(id: id, revision: revision),
        progressHandler: progressHandler
    )
}

internal func loadModel(
    hub: HubApi = defaultHubApi,
    directory: URL,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await loadModel(
        hub: hub,
        configuration: ModelConfiguration(directory: directory),
        progressHandler: progressHandler
    )
}

internal func loadModelContainer(
    hub: HubApi = defaultHubApi,
    directory: URL,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await loadModelContainer(
        hub: hub,
        configuration: ModelConfiguration(directory: directory),
        progressHandler: progressHandler
    )
}

internal protocol ModelFactoryTrampoline {
    static func modelFactory() -> ModelFactory?
}

final internal class ModelFactoryRegistry: @unchecked Sendable {
    internal typealias Trampoline = @Sendable () -> ModelFactory?

    internal static let shared = ModelFactoryRegistry()

    private static let defaultTrampolines: [Trampoline] = [
        {
            (NSClassFromString("MLXVLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                .modelFactory()
        },
        {
            (NSClassFromString("MLXLLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                .modelFactory()
        }
    ]

    private let lock = NSLock()
    private var trampolines: [Trampoline]

    internal init(trampolines: [Trampoline] = defaultTrampolines) {
        self.trampolines = trampolines
    }

    internal func addTrampoline(_ trampoline: @escaping Trampoline) {
        lock.withLock {
            trampolines.append(trampoline)
        }
    }

    internal func modelFactories() -> [ModelFactory] {
        lock.withLock {
            trampolines.compactMap { $0() }
        }
    }
}
