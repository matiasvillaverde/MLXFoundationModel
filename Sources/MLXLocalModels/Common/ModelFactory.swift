// Copyright © 2024 Apple Inc.

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
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .unsupportedProcessorType(let type):
            return "Unsupported processor type: \(type)"
        case .noModelFactoryAvailable:
            return "No model factory available via ModelFactoryRegistry"
        case .configurationDecodingError(let file, let modelName, let decodingError):
            let errorDetail = extractDecodingErrorDetail(decodingError)
            return "Failed to parse \(file) for model '\(modelName)': \(errorDetail)"
        }
    }

    private func extractDecodingErrorDetail(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, let context):
            let path = (context.codingPath + [key]).map { $0.stringValue }.joined(separator: ".")
            return "Missing field '\(path)'"
        case .typeMismatch(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Type mismatch at '\(path)'"
        case .valueNotFound(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Missing value at '\(path)'"
        case .dataCorrupted(let context):
            if context.codingPath.isEmpty {
                return "Invalid JSON"
            } else {
                let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
                return "Invalid data at '\(path)'"
            }
        @unknown default:
            return error.localizedDescription
        }
    }
}

/// Context of types that work together to provide a ``LanguageModel``.
///
/// A ``ModelContext`` is created by ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This contains the following:
///
/// - ``ModelConfiguration`` -- identifier for the model
/// - ``LanguageModel`` -- the model itself, see ``generate(input:parameters:context:didGenerate:)``
/// - `Tokenizer` -- the tokenizer used to convert text into tokens
///
/// This type is marked `@unchecked Sendable` because:
/// - It contains `LanguageModel` (a protocol) which may have non-Sendable implementations
/// - It contains `Tokenizer` which is NOT inherently Sendable
/// - However, instances are used in a controlled context (ModelContainer actor)
/// - The model and tokenizer are only accessed from the ModelContainer's serial executor
///
/// Safety guarantees:
/// - Actor isolation: Always accessed through `ModelContainer.perform` which ensures serial execution
/// - Single-threaded GPU access: The model operations run on a single thread (GPU constraint)
/// - No concurrent modification: Only one operation can execute at a time per ModelContainer
/// - Controlled lifecycle: Created once during model loading, used serially for generation
///
/// See also ``ModelFactory/loadContainer(hub:configuration:progressHandler:)`` and
/// ``ModelContainer``.
internal struct ModelContext: @unchecked Sendable {
    internal var configuration: ModelConfiguration
    internal var model: any LanguageModel
    internal var tokenizer: Tokenizer
    internal var grammarCompiler: GrammarConstraintCompiler?
    internal var grammarCompilerError: Error?

    internal init(
        configuration: ModelConfiguration, model: any LanguageModel,
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

    /// Convert a text prompt into tokenized input for the model.
    ///
    /// Rendered ContextBuilder prompts are encoded exactly as provided.
    /// Plain prompts use the tokenizer's native chat template when the tokenizer declares one.
    internal func tokenize(input: LLMInput) throws -> LMInput {
        try PromptTokenizer.tokenize(input: input, tokenizer: tokenizer)
    }

    /// This method preserves the legacy plain-prompt tokenization path.
    ///
    /// - Parameter prompt: The text prompt to tokenize
    /// - Returns: Tokenized input ready for model inference
    /// - Throws: Errors from the tokenizer
    internal func tokenize(prompt: String) throws -> LMInput {
        try PromptTokenizer.tokenize(prompt: prompt, tokenizer: tokenizer)
    }
}

internal enum PromptTokenizer {
    internal static func tokenize(
        input: LLMInput,
        tokenizer: Tokenizer
    ) throws -> LMInput {
        guard input.promptMetadata == nil, input.promptCacheIdentity == nil else {
            return encode(prompt: input.context, tokenizer: tokenizer)
        }
        return try tokenize(prompt: input.context, tokenizer: tokenizer)
    }

    internal static func tokenize(
        prompt: String,
        tokenizer: Tokenizer
    ) throws -> LMInput {
        if PromptFormatDetector.isPreformatted(prompt) {
            return encode(prompt: prompt, tokenizer: tokenizer)
        }

        guard tokenizer.hasChatTemplate else {
            return encode(prompt: prompt, tokenizer: tokenizer)
        }

        let messages: [Message] = [
            ["role": "user", "content": prompt]
        ]
        let tokens = try tokenizer.applyChatTemplate(messages: messages)
        return LMInput(tokens: MLXArray(tokens))
    }

    private static func encode(
        prompt: String,
        tokenizer: Tokenizer
    ) -> LMInput {
        LMInput(tokens: MLXArray(tokenizer.encode(text: prompt)))
    }
}

/// Protocol for code that can load models.
///
/// ## See Also
/// - ``loadModel(hub:id:progressHandler:)``
/// - ``loadModel(hub:directory:progressHandler:)``
/// - ``loadModelContainer(hub:id:progressHandler:)``
/// - ``loadModelContainer(hub:directory:progressHandler:)``
internal protocol ModelFactory: Sendable {

    var modelRegistry: AbstractModelRegistry { get }

    func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext

    func _loadContainer(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContainer

}

extension ModelFactory {

    /// Resolve a model identifier, e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit", into
    /// a ``ModelConfiguration``.
    ///
    /// This will either create a new (mostly unconfigured) ``ModelConfiguration`` or
    /// return a registered instance that matches the id.
    ///
    /// - Note: If the id doesn't exists in the configuration, this will return a new instance of it.
    /// If you want to check if the configuration in model registry, you should use ``contains(id:)``.
    internal func configuration(id: String) -> ModelConfiguration {
        modelRegistry.configuration(id: id)
    }

    /// Returns true if ``modelRegistry`` contains a model with the id. Otherwise, false.
    internal func contains(id: String) -> Bool {
        modelRegistry.contains(id: id)
    }

}

/// Default instance of HubApi to use.  This is configured to save downloads into the caches directory.
internal nonisolated(unsafe) var defaultHubApi: HubApi = {
    HubApi(downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first)
}()

extension ModelFactory {

    /// Load a model identified by a ``ModelConfiguration`` and produce a ``ModelContext``.
    ///
    /// This method returns a ``ModelContext``. See also
    /// ``loadContainer(hub:configuration:progressHandler:)`` for a method that
    /// returns a ``ModelContainer``.
    ///
    /// ## See Also
    /// - ``loadModel(hub:id:progressHandler:)``
    /// - ``loadModelContainer(hub:id:progressHandler:)``
    internal func load(
        hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ModelContext {
        try await _load(hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    /// Load a model identified by a ``ModelConfiguration`` and produce a ``ModelContainer``.
    internal func loadContainer(
        hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        try await _loadContainer(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    internal func _loadContainer(
        hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let context = try await _load(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
        return ModelContainer(context: context)
    }

}

/// Load a model given a ``ModelConfiguration``.
///
/// This will load and return a ``ModelContext``.  This holds the model and tokenzier without
/// an `actor` providing an isolation context.  Use this call when you control the isolation context
/// and can hold the ``ModelContext`` directly.
///
/// - Parameters:
///   - hub: optional HubApi -- by default uses ``defaultHubApi``
///   - configuration: a ``ModelConfiguration``
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
internal func loadModel(
    hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(hub: hub, configuration: configuration, progressHandler: progressHandler)
    }
}

/// Load a model given a ``ModelConfiguration``.
///
/// This will load and return a ``ModelContainer``.  This holds a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - hub: optional HubApi -- by default uses ``defaultHubApi``
///   - configuration: a ``ModelConfiguration``
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
internal func loadModelContainer(
    hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
    }
}

/// Load a model given a huggingface identifier.
///
/// This will load and return a ``ModelContext``.  This holds the model and tokenzier without
/// an `actor` providing an isolation context.  Use this call when you control the isolation context
/// and can hold the ``ModelContext`` directly.
///
/// - Parameters:
///   - hub: optional HubApi -- by default uses ``defaultHubApi``
///   - id: huggingface model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
internal func loadModel(
    hub: HubApi = defaultHubApi, id: String, revision: String = "main",
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            hub: hub, configuration: .init(id: id, revision: revision),
            progressHandler: progressHandler)
    }
}

/// Load a model given a huggingface identifier.
///
/// This will load and return a ``ModelContainer``.  This holds a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - hub: optional HubApi -- by default uses ``defaultHubApi``
///   - id: huggingface model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
internal func loadModelContainer(
    hub: HubApi = defaultHubApi, id: String, revision: String = "main",
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            hub: hub, configuration: .init(id: id, revision: revision),
            progressHandler: progressHandler)
    }
}

/// Load a model given a directory of configuration and weights.
///
/// This will load and return a ``ModelContext``.  This holds the model and tokenzier without
/// an `actor` providing an isolation context.  Use this call when you control the isolation context
/// and can hold the ``ModelContext`` directly.
///
/// - Parameters:
///   - hub: optional HubApi -- by default uses ``defaultHubApi``
///   - directory: directory of configuration and weights
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
internal func loadModel(
    hub: HubApi = defaultHubApi, directory: URL,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            hub: hub, configuration: .init(directory: directory), progressHandler: progressHandler)
    }
}

/// Load a model given a directory of configuration and weights.
///
/// This will load and return a ``ModelContainer``.  This holds a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - hub: optional HubApi -- by default uses ``defaultHubApi``
///   - directory: directory of configuration and weights
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
internal func loadModelContainer(
    hub: HubApi = defaultHubApi, directory: URL,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            hub: hub, configuration: .init(directory: directory), progressHandler: progressHandler)
    }
}

private func load<R>(loader: (ModelFactory) async throws -> sending R) async throws -> sending R {
    let factories = ModelFactoryRegistry.shared.modelFactories()
    var lastError: Error?
    for factory in factories {
        do {
            let model = try await loader(factory)
            return model
        } catch {
            lastError = error
        }
    }

    if let lastError {
        throw lastError
    } else {
        throw ModelFactoryError.noModelFactoryAvailable
    }
}

/// Protocol for types that can provide ModelFactory instances.
///
/// Not used directly.
///
/// This is used internally to provide dynamic lookup of a trampoline -- this lets
/// API in MLXLMCommon use code present in MLXLLM:
///
/// ```swift
/// public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
///     public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
///         LLMModelFactory.shared
///     }
/// }
/// ```
///
/// That is looked up dynamically with:
///
/// ```swift
/// {
///     (NSClassFromString("MLXVLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
///         .modelFactory()
/// }
/// ```
///
/// ## See Also
/// - ``ModelFactoryRegistry``
internal protocol ModelFactoryTrampoline {
    static func modelFactory() -> ModelFactory?
}

/// Registry of ``ModelFactory`` trampolines.
///
/// This allows ``loadModel(hub:id:progressHandler:)`` to use any ``ModelFactory`` instances
/// available but be defined in the `LLMCommon` layer.  This is not typically used directly -- it is
/// called via ``loadModel(hub:id:progressHandler:)``:
///
/// ```swift
/// let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
/// ```
///
/// This class is marked `@unchecked Sendable` because:
/// - It contains mutable state (`trampolines` array) that is protected by explicit synchronization (`NSLock`)
/// - All mutations to the trampoline list are guarded by lock-protected critical sections
/// - The lock ensures that all array accesses are properly serialized
///
/// Safety guarantees:
/// - Atomic operations: All reads and writes to `trampolines` are protected by NSLock
/// - Thread-safe registration: Model factories can be registered from any thread
/// - Thread-safe retrieval: Multiple threads can safely query available factories
/// - No data races: The lock serializes all access to the mutable array
///
/// ## See Also
/// - ``loadModel(hub:id:progressHandler:)``
/// - ``loadModel(hub:directory:progressHandler:)``
/// - ``loadModelContainer(hub:id:progressHandler:)``
/// - ``loadModelContainer(hub:directory:progressHandler:)``
final internal class ModelFactoryRegistry: @unchecked Sendable {
    internal static let shared = ModelFactoryRegistry()

    private let lock = NSLock()
    private var trampolines: [() -> ModelFactory?]

    private init() {
        self.trampolines = [
            {
                (NSClassFromString("MLXVLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
            {
                (NSClassFromString("MLXLLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
        ]
    }

    internal func addTrampoline(_ trampoline: @escaping () -> ModelFactory?) {
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
