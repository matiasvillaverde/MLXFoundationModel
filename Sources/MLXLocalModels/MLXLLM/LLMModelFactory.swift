// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX

import Tokenizers

private func decodeConfiguration<C: Codable & Sendable>(from url: URL) throws -> C {
    try JSONDecoder.json5().decode(C.self, from: Data(contentsOf: url))
}

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable & Sendable, M>(
    _: C.Type,
    _ modelInit: @escaping @Sendable (C) -> M
) -> @Sendable (URL) throws -> M {
    { url in
        let configuration: C = try decodeConfiguration(from: url)
        return modelInit(configuration)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(hub:configuration:progressHandler:)``.
///
/// This class is marked `@unchecked Sendable` because:
/// - It inherits from `ModelTypeRegistry`, which is marked `@unchecked Sendable`
/// - The parent class handles all thread safety through NSLock synchronization
/// - This class is immutable after initialization (creators map is set once)
///
/// Safety guarantees:
/// - Immutable after creation: The shared instance is initialized once with a fixed set of model creators
/// - Thread-safe registration: Inherits NSLock-protected operations from `ModelTypeRegistry`
/// - Safe concurrent access: Multiple threads can safely query model creators
internal class LLMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {
    /// Shared instance with default model types.
    public static let shared: LLMTypeRegistry = .init(creators: all())

    /// All predefined model types.
    private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] {
        [
            "mistral": create(LlamaConfiguration.self, LlamaModel.init),
            "llama": create(LlamaConfiguration.self, LlamaModel.init),
            "phi": create(PhiConfiguration.self, PhiModel.init),
            "phi3": create(Phi3Configuration.self, Phi3Model.init),
            "phimoe": create(PhiMoEConfiguration.self, PhiMoEModel.init),
            "gemma": create(GemmaConfiguration.self, GemmaModel.init),
            "gemma2": create(Gemma2Configuration.self, Gemma2Model.init),
            "gemma3": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "gemma3_text": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "gemma3n": create(Gemma3nTextConfiguration.self, Gemma3nTextModel.init),
            "gemma4": create(Gemma4TextConfiguration.self, Gemma4Model.init),
            "gemma4_text": create(Gemma4TextConfiguration.self, Gemma4Model.init),
            "qwen2": create(Qwen2Configuration.self, Qwen2Model.init),
            "qwen3": create(Qwen3Configuration.self, Qwen3Model.init),
            "qwen3_moe": create(Qwen3MoEConfiguration.self, Qwen3MoEModel.init),
            "qwen3_next": create(Qwen3NextConfiguration.self, Qwen3NextModel.init),
            "qwen3_5": create(Qwen35Configuration.self, Qwen35Model.init),
            "qwen3_5_moe": create(Qwen35Configuration.self, Qwen35MoEModel.init),
            "qwen3_5_text": create(Qwen35TextConfiguration.self, Qwen35TextModel.init),
            "minicpm": create(MiniCPMConfiguration.self, MiniCPMModel.init),
            "starcoder2": create(Starcoder2Configuration.self, Starcoder2Model.init),
            "cohere": create(CohereConfiguration.self, CohereModel.init),
            "openelm": create(OpenElmConfiguration.self, OpenELMModel.init),
            "internlm2": create(InternLM2Configuration.self, InternLM2Model.init),
            "deepseek_v3": create(DeepseekV3Configuration.self, DeepseekV3Model.init),
            "granite": create(GraniteConfiguration.self, GraniteModel.init),
            "granitemoehybrid": create(
                GraniteMoeHybridConfiguration.self,
                GraniteMoeHybridModel.init
            ),
            "mimo": create(MiMoConfiguration.self, MiMoModel.init),
            "mimo_v2_flash": create(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init),
            "minimax": create(MiniMaxConfiguration.self, MiniMaxModel.init),
            "glm4": create(GLM4Configuration.self, GLM4Model.init),
            "glm4_moe": create(GLM4MoEConfiguration.self, GLM4MoEModel.init),
            "glm4_moe_lite": create(GLM4MoELiteConfiguration.self, GLM4MoELiteModel.init),
            "acereason": create(Qwen2Configuration.self, Qwen2Model.init),
            "falcon_h1": create(FalconH1Configuration.self, FalconH1Model.init),
            "bitnet": create(BitnetConfiguration.self, BitnetModel.init),
            "smollm3": create(SmolLM3Configuration.self, SmolLM3Model.init),
            "ernie4_5": create(Ernie45Configuration.self, Ernie45Model.init),
            "lfm2": create(LFM2Configuration.self, LFM2Model.init),
            "baichuan_m1": create(BaichuanM1Configuration.self, BaichuanM1Model.init),
            "exaone4": create(Exaone4Configuration.self, Exaone4Model.init),
            "gpt_oss": create(GPTOSSConfiguration.self, GPTOSSModel.init),
            "lille-130m": create(Lille130mConfiguration.self, Lille130mModel.init),
            "olmoe": create(OlmoEConfiguration.self, OlmoEModel.init),
            "olmo2": create(Olmo2Configuration.self, Olmo2Model.init),
            "olmo3": create(Olmo3Configuration.self, Olmo3Model.init),
            "bailing_moe": create(BailingMoeConfiguration.self, BailingMoeModel.init),
            "lfm2_moe": create(LFM2MoEConfiguration.self, LFM2MoEModel.init),
            "nanochat": create(NanoChatConfiguration.self, NanoChatModel.init),
            "nemotron_h": create(NemotronHConfiguration.self, NemotronHModel.init),
            "afmoe": create(AfMoEConfiguration.self, AfMoEModel.init),
            "jamba_3b": create(JambaConfiguration.self, JambaModel.init),
            "mistral3": create(Mistral3TextConfiguration.self, Mistral3TextModel.init),
            "apertus": create(ApertusConfiguration.self, ApertusModel.init)
        ]
    }
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The Python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
///
/// This class is marked `@unchecked Sendable` because:
/// - It inherits from `AbstractModelRegistry`, which is marked `@unchecked Sendable`
/// - The parent class handles all thread safety through NSLock synchronization
/// - This class provides predefined model configurations (immutable after initialization)
///
/// Safety guarantees:
/// - Immutable configurations: The shared instance is initialized once with predefined models
/// - Thread-safe access: Inherits NSLock-protected operations from `AbstractModelRegistry`
/// - Safe concurrent queries: Multiple threads can safely retrieve model configurations
internal class LLMRegistry: AbstractModelRegistry, @unchecked Sendable {
    /// Shared instance with default model configurations.
    public static let shared = LLMRegistry(modelConfigurations: all())

    public static let smolLM135M4bit = ModelConfiguration(
        id: "mlx-community/SmolLM-135M-Instruct-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    public static let mistralNeMo4bit = ModelConfiguration(
        id: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        defaultPrompt: "Explain quaternions."
    )

    public static let mistral7B4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        defaultPrompt: "Describe the Swift language."
    )

    public static let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    )

    public static let deepSeekR1SevenB4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        defaultPrompt: "Is 9.9 greater or 9.11?"
    )

    public static let phi4bit = ModelConfiguration(
        id: "mlx-community/phi-2-hf-4bit-mlx",
        // https://www.promptingguide.ai/models/phi-2
        defaultPrompt: "Why is the sky blue?"
    )

    public static let phi3Point5Four4bit = ModelConfiguration(
        id: "mlx-community/Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    public static let phi3Point5MoE = ModelConfiguration(
        id: "mlx-community/Phi-3.5-MoE-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    ) {
        prompt in
        "<|user|>\n\(prompt)<|end|>\n<|assistant|>\n"
    }

    public static let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "what is the difference between lettuce and cabbage?"
    )

    public static let gemma2Nine9bIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-9b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    public static let gemma2Two2bIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-2b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    public static let gemma3One1BQat4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-1b-it-qat-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<end_of_turn>"]
    )

    public static let gemma3nE4BItLmBf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    public static let gemma3nE2BItLmBf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    public static let gemma3nE4BItLm4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    public static let gemma3nE2BItLm4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    public static let gemma4E2BIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-e2b-it-4bit",
        defaultPrompt: "Write one short sentence about private on-device AI.",
        extraEOSTokens: ["<turn|>"]
    )

    public static let gemma4E4BIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-e4b-it-4bit",
        defaultPrompt: "Write one short sentence about private on-device AI.",
        extraEOSTokens: ["<turn|>"]
    )

    public static let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "why is the sky blue?"
    )

    public static let qwen2Point5Seven7b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let qwen2Point5One1Point5b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let qwen3Zero0Point6b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-0.6B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let qwen3One1Point7b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-1.7B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let qwen3Four4b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-4B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let qwen3Eight8b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-8B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let qwen3MoE30bA3b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-30B-A3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let openelm270m4bit = ModelConfiguration(
        id: "mlx-community/OpenELM-270M-Instruct",
        // https://huggingface.co/apple/OpenELM
        defaultPrompt: "Once upon a time there was"
    )

    public static let llama3Point1Eight8B4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    public static let llama3Eight8B4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    public static let llama3Point2One1B4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    public static let llama3Point2Three3B4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    public static let deepseekR1Four4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    public static let granite3Point3Two2b4bit = ModelConfiguration(
        id: "mlx-community/granite-3.3-2b-instruct-4bit",
        defaultPrompt: ""
    )

    public static let mimo7bSft4bit = ModelConfiguration(
        id: "mlx-community/MiMo-7B-SFT-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let glm4Nine9b4bit = ModelConfiguration(
        id: "mlx-community/GLM-4-9B-0414-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let acereason7b4bit = ModelConfiguration(
        id: "mlx-community/AceReason-Nemotron-7B-4bit",
        defaultPrompt: ""
    )

    public static let bitnetB1Point58Two2b4t4bit = ModelConfiguration(
        id: "mlx-community/bitnet-b1.58-2B-4T-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let baichuanM1Fourteen14bInstruct4bit = ModelConfiguration(
        id: "mlx-community/Baichuan-M1-14B-Instruct-4bit-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let smollm3Three3b4bit = ModelConfiguration(
        id: "mlx-community/SmolLM3-3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let ernie45Zero0Point3BPTBf16Ft = ModelConfiguration(
        id: "mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let lfm2One1Point2b4bit = ModelConfiguration(
        id: "mlx-community/LFM2-1.2B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    public static let exaone4Point0One1Point2b4bit = ModelConfiguration(
        id: "mlx-community/exaone-4.0-1.2b-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    private static func all() -> [ModelConfiguration] {
        [
            codeLlama13b4bit,
            deepSeekR1SevenB4bit,
            gemma2bQuantized,
            gemma2Two2bIt4bit,
            gemma2Nine9bIt4bit,
            gemma3One1BQat4bit,
            gemma3nE4BItLmBf16,
            gemma3nE2BItLmBf16,
            gemma3nE4BItLm4bit,
            gemma3nE2BItLm4bit,
            gemma4E2BIt4bit,
            gemma4E4BIt4bit,
            granite3Point3Two2b4bit,
            llama3Point1Eight8B4bit,
            llama3Point2One1B4bit,
            llama3Point2Three3B4bit,
            llama3Eight8B4bit,
            mistral7B4bit,
            mistralNeMo4bit,
            openelm270m4bit,
            phi3Point5MoE,
            phi3Point5Four4bit,
            phi4bit,
            qwen205b4bit,
            qwen2Point5Seven7b,
            qwen2Point5One1Point5b,
            qwen3Zero0Point6b4bit,
            qwen3One1Point7b4bit,
            qwen3Four4b4bit,
            qwen3Eight8b4bit,
            qwen3MoE30bA3b4bit,
            smolLM135M4bit,
            deepseekR1Four4bit,
            mimo7bSft4bit,
            glm4Nine9b4bit,
            acereason7b4bit,
            bitnetB1Point58Two2b4t4bit,
            smollm3Three3b4bit,
            ernie45Zero0Point3BPTBf16Ft,
            lfm2One1Point2b4bit,
            baichuanM1Fourteen14bInstruct4bit,
            exaone4Point0One1Point2b4bit
        ]
    }
}

@available(*, deprecated, renamed: "LLMRegistry", message: "Please use LLMRegistry directly.")
internal typealias ModelRegistry = LLMRegistry

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await LLMModelFactory.shared.loadContainer(
///     configuration: LLMRegistry.llama3_8B_4bit)
/// ```
internal final class LLMModelFactory: ModelFactory {
    public init(typeRegistry: ModelTypeRegistry, modelRegistry: AbstractModelRegistry) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared)

    /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext {
        // Create progress tracker
        let progress = Progress(totalUnitCount: 100)

        // Step 1: Download (0-30%)
        progressHandler(progress)
        let modelDirectory = try await downloadModel(
            hub: hub,
            configuration: configuration) { _ in
                progress.completedUnitCount = 30
                progressHandler(progress)
        }

        // Step 2: Load config and create model (30-50%)
        // load the generic config to understand which model and how to load the weights
        let configurationURL = modelDirectory.appending(component: "config.json")

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(
                BaseConfiguration.self, from: Data(contentsOf: configurationURL))
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: LanguageModel
        do {
            model = try typeRegistry.createModel(
                configuration: configurationURL, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        progress.completedUnitCount = 50
        progressHandler(progress)

        var resolvedConfiguration = configuration
        resolvedConfiguration.eosTokenIds = loadEOSTokenIds(
            baseConfig: baseConfig,
            modelDirectory: modelDirectory
        )

        async let tokenizerTask = loadTokenizer(configuration: configuration, hub: hub)

        // Step 3: Load weights (50-80%)
        // apply the weights to the bare model
        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        progress.completedUnitCount = 80
        progressHandler(progress)

        // Step 4: Load tokenizer (80-100%)
        let tokenizer = try await tokenizerTask
        var grammarStopTokenIds = resolvedConfiguration.eosTokenIds
        if let eosTokenId = tokenizer.eosTokenId {
            grammarStopTokenIds.insert(eosTokenId)
        }
        if let unknownTokenId = tokenizer.unknownTokenId {
            grammarStopTokenIds.insert(unknownTokenId)
        }
        grammarStopTokenIds.formUnion(
            resolvedConfiguration.extraEOSTokens
                .compactMap { tokenizer.convertTokenToId($0) }
        )
        let grammarCompiler: GrammarConstraintCompiler?
        let grammarCompilerError: Error?
        do {
            grammarCompiler = try GrammarConstraintCompiler(
                modelDirectory: modelDirectory,
                stopTokenIds: grammarStopTokenIds
            )
            grammarCompilerError = nil
        } catch {
            grammarCompiler = nil
            grammarCompilerError = error
        }

        progress.completedUnitCount = 100
        progressHandler(progress)

        return .init(
            configuration: resolvedConfiguration,
            model: model,
            tokenizer: tokenizer,
            grammarCompiler: grammarCompiler,
            grammarCompilerError: grammarCompilerError
        )
    }

    private func loadEOSTokenIds(
        baseConfig: BaseConfiguration,
        modelDirectory: URL
    ) -> Set<Int> {
        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        guard
            let data = try? Data(contentsOf: generationConfigURL),
            let generationConfig = try? JSONDecoder.json5().decode(
                GenerationConfigFile.self,
                from: data
            ),
            let generationEOSTokenIds = generationConfig.eosTokenIds?.values
        else {
            return Set(baseConfig.eosTokenIds?.values ?? [])
        }

        return Set(generationEOSTokenIds)
    }
}

internal class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any ModelFactory)? {
        LLMModelFactory.shared
    }
}
