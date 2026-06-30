import Foundation
import Hub
import MLX
import Tokenizers

private typealias LanguageModelCreator = @Sendable (URL) throws -> any LanguageModel

private struct LLMModelTypeRegistration: Sendable {
    let names: [String]
    let makeModel: LanguageModelCreator
}

private func modelType<C: Codable & Sendable>(
    _ names: String...,
    configuration: C.Type,
    makeModel: @escaping @Sendable (C) -> any LanguageModel
) -> LLMModelTypeRegistration {
    LLMModelTypeRegistration(names: names) { url in
        let data = try Data(contentsOf: url)
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return makeModel(configuration)
    }
}

internal class LLMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {
    public static let shared: LLMTypeRegistry = .init(creators: defaultCreators())

    private static func defaultCreators() -> [String: LanguageModelCreator] {
        Dictionary(uniqueKeysWithValues: defaultTypeRegistrations().flatMap { registration in
            registration.names.map { ($0, registration.makeModel) }
        })
    }

    private static func defaultTypeRegistrations() -> [LLMModelTypeRegistration] {
        [
            modelType("mistral", "llama", configuration: LlamaConfiguration.self) {
                LlamaModel($0)
            },
            modelType("phi", configuration: PhiConfiguration.self) { PhiModel($0) },
            modelType("phi3", configuration: Phi3Configuration.self) { Phi3Model($0) },
            modelType("phimoe", configuration: PhiMoEConfiguration.self) { PhiMoEModel($0) },
            modelType("gemma", configuration: GemmaConfiguration.self) { GemmaModel($0) },
            modelType("gemma2", configuration: Gemma2Configuration.self) { Gemma2Model($0) },
            modelType("gemma3", "gemma3_text", configuration: Gemma3TextConfiguration.self) {
                Gemma3TextModel($0)
            },
            modelType("gemma3n", configuration: Gemma3nTextConfiguration.self) {
                Gemma3nTextModel(config: $0)
            },
            modelType("gemma4", "gemma4_text", configuration: Gemma4TextConfiguration.self) {
                Gemma4Model($0)
            },
            modelType(
                "gemma4_assistant",
                "gemma4_unified_assistant",
                configuration: Gemma4AssistantConfiguration.self
            ) {
                Gemma4AssistantModel($0)
            },
            modelType("qwen2", configuration: Qwen2Configuration.self) { Qwen2Model($0) },
            modelType("qwen2_moe", configuration: Qwen2MoEConfiguration.self) {
                Qwen2MoEModel($0)
            },
            modelType("qwen3", configuration: Qwen3Configuration.self) { Qwen3Model($0) },
            modelType("qwen3_moe", configuration: Qwen3MoEConfiguration.self) {
                Qwen3MoEModel($0)
            },
            modelType("qwen3_next", configuration: Qwen3NextConfiguration.self) {
                Qwen3NextModel($0)
            },
            modelType("qwen3_5", configuration: Qwen35Configuration.self) { Qwen35Model($0) },
            modelType("qwen3_5_moe", configuration: Qwen35Configuration.self) {
                Qwen35MoEModel($0)
            },
            modelType("qwen3_5_text", configuration: Qwen35TextConfiguration.self) {
                Qwen35TextModel($0)
            },
            modelType("minicpm", configuration: MiniCPMConfiguration.self) { MiniCPMModel($0) },
            modelType("minicpm3", configuration: MiniCPM3Configuration.self) {
                MiniCPM3Model($0)
            },
            modelType("starcoder2", configuration: Starcoder2Configuration.self) {
                Starcoder2Model($0)
            },
            modelType("cohere", configuration: CohereConfiguration.self) { CohereModel($0) },
            modelType("openelm", configuration: OpenElmConfiguration.self) { OpenELMModel($0) },
            modelType("internlm2", configuration: InternLM2Configuration.self) {
                InternLM2Model($0)
            },
            modelType("deepseek_v2", configuration: DeepseekV2Configuration.self) {
                DeepseekV2Model($0)
            },
            modelType("deepseek_v3", configuration: DeepseekV3Configuration.self) {
                DeepseekV3Model($0)
            },
            modelType("deepseek_v32", configuration: DeepseekV32Configuration.self) {
                DeepseekV32Model($0)
            },
            modelType("granite", configuration: GraniteConfiguration.self) { GraniteModel($0) },
            modelType("granitemoe", configuration: GraniteMoEConfiguration.self) {
                GraniteMoEModel($0)
            },
            modelType("granitemoehybrid", configuration: GraniteMoeHybridConfiguration.self) {
                GraniteMoeHybridModel($0)
            },
            modelType("mimo", configuration: MiMoConfiguration.self) { MiMoModel($0) },
            modelType("mimo_v2_flash", configuration: MiMoV2FlashConfiguration.self) {
                MiMoV2FlashModel($0)
            },
            modelType("minimax", configuration: MiniMaxConfiguration.self) { MiniMaxModel($0) },
            modelType("glm4", configuration: GLM4Configuration.self) { GLM4Model($0) },
            modelType("glm4_moe", configuration: GLM4MoEConfiguration.self) {
                GLM4MoEModel($0)
            },
            modelType(
                "glm4_moe_lite",
                "glm_moe_dsa",
                configuration: GLM4MoELiteConfiguration.self
            ) {
                GLM4MoELiteModel($0)
            },
            modelType("acereason", configuration: Qwen2Configuration.self) { Qwen2Model($0) },
            modelType("falcon_h1", configuration: FalconH1Configuration.self) {
                FalconH1Model($0)
            },
            modelType("bitnet", configuration: BitnetConfiguration.self) { BitnetModel($0) },
            modelType("smollm3", configuration: SmolLM3Configuration.self) { SmolLM3Model($0) },
            modelType("ernie4_5", configuration: Ernie45Configuration.self) { Ernie45Model($0) },
            modelType("helium", configuration: HeliumConfiguration.self) { HeliumModel($0) },
            modelType("lfm2", configuration: LFM2Configuration.self) { LFM2Model($0) },
            modelType("mamba", configuration: MambaConfiguration.self) { MambaModel($0) },
            modelType("mamba2", configuration: Mamba2Configuration.self) { Mamba2Model($0) },
            modelType("baichuan_m1", configuration: BaichuanM1Configuration.self) {
                BaichuanM1Model($0)
            },
            modelType("exaone", configuration: ExaoneConfiguration.self) { ExaoneModel($0) },
            modelType("exaone4", configuration: Exaone4Configuration.self) { Exaone4Model($0) },
            modelType("gpt_oss", configuration: GPTOSSConfiguration.self) { GPTOSSModel($0) },
            modelType("gpt2", configuration: GPT2Configuration.self) { GPT2Model($0) },
            modelType("gpt_bigcode", configuration: GPTBigCodeConfiguration.self) {
                GPTBigCodeModel($0)
            },
            modelType("gpt_neox", configuration: GPTNeoXConfiguration.self) {
                GPTNeoXModel($0)
            },
            modelType("lille-130m", configuration: Lille130mConfiguration.self) {
                Lille130mModel($0)
            },
            modelType("olmo", configuration: OlmoConfiguration.self) { OlmoModel($0) },
            modelType("olmoe", configuration: OlmoEConfiguration.self) { OlmoEModel($0) },
            modelType("olmo2", configuration: Olmo2Configuration.self) { Olmo2Model($0) },
            modelType("olmo3", configuration: Olmo3Configuration.self) { Olmo3Model($0) },
            modelType("bailing_moe", configuration: BailingMoeConfiguration.self) {
                BailingMoeModel($0)
            },
            modelType("stablelm", configuration: StableLMConfiguration.self) {
                StableLMModel($0)
            },
            modelType("lfm2_moe", configuration: LFM2MoEConfiguration.self) {
                LFM2MoEModel($0)
            },
            modelType("nanochat", configuration: NanoChatConfiguration.self) {
                NanoChatModel($0)
            },
            modelType("nemotron_h", configuration: NemotronHConfiguration.self) {
                NemotronHModel($0)
            },
            modelType("afmoe", configuration: AfMoEConfiguration.self) { AfMoEModel($0) },
            modelType("jamba", "jamba_3b", configuration: JambaConfiguration.self) {
                JambaModel($0)
            },
            modelType("mistral3", "ministral3", configuration: Mistral3TextConfiguration.self) {
                Mistral3TextModel($0)
            },
            modelType("apertus", configuration: ApertusConfiguration.self) {
                ApertusModel($0)
            }
        ]
    }
}

internal struct LLMGenerationTokenConfig: Equatable, Sendable {
    var eosTokenIds: Set<Int>
    var suppressTokenIds: Set<Int>

    static func load(baseConfig: BaseConfiguration, modelDirectory: URL) -> Self {
        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        guard
            let data = try? Data(contentsOf: generationConfigURL),
            let generationConfig = try? JSONDecoder.json5().decode(
                GenerationConfigFile.self,
                from: data
            )
        else {
            return .init(
                eosTokenIds: Set(baseConfig.eosTokenIds?.values ?? []),
                suppressTokenIds: []
            )
        }

        let eosTokenIds = generationConfig.eosTokenIds?.values ?? baseConfig.eosTokenIds?.values ?? []
        return .init(
            eosTokenIds: Set(eosTokenIds),
            suppressTokenIds: Set(generationConfig.suppressTokenIds?.values ?? [])
        )
    }
}

internal class LLMRegistry: AbstractModelRegistry, @unchecked Sendable {
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
    )

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

    public static let deepseekV2LiteChat4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx",
        defaultPrompt: "Write one short sentence about efficient expert models."
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
            deepseekV2LiteChat4bit,
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

internal final class LLMModelFactory: ModelFactory {
    public init(typeRegistry: ModelTypeRegistry, modelRegistry: AbstractModelRegistry) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared,
        modelRegistry: LLMRegistry.shared
    )

    public let typeRegistry: ModelTypeRegistry

    public let modelRegistry: AbstractModelRegistry

    public func _load(
        hub: HubApi,
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext {
        let progress = Progress(totalUnitCount: 100)

        progressHandler(progress)
        let modelDirectory = try await downloadModel(
            hub: hub,
            configuration: configuration
        ) { _ in
            progress.completedUnitCount = 30
            progressHandler(progress)
        }

        let configurationURL = modelDirectory.appending(component: "config.json")
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(
                BaseConfiguration.self,
                from: Data(contentsOf: configurationURL)
            )
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent,
                configuration.name,
                error
            )
        }

        let model: LanguageModel
        do {
            model = try typeRegistry.createModel(
                configuration: configurationURL,
                modelType: baseConfig.modelType
            )
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent,
                configuration.name,
                error
            )
        }

        progress.completedUnitCount = 50
        progressHandler(progress)

        var resolvedConfiguration = configuration
        let generationTokenConfig = LLMGenerationTokenConfig.load(
            baseConfig: baseConfig,
            modelDirectory: modelDirectory
        )
        resolvedConfiguration.eosTokenIds = generationTokenConfig.eosTokenIds
        resolvedConfiguration.suppressTokenIds = generationTokenConfig.suppressTokenIds

        async let tokenizerTask = loadTokenizer(configuration: configuration, hub: hub)

        try loadWeights(
            modelDirectory: modelDirectory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization
        )

        progress.completedUnitCount = 80
        progressHandler(progress)

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
}

internal class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any ModelFactory)? {
        LLMModelFactory.shared
    }
}
