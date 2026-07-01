import Testing

@Suite("MLX real-model catalog")
struct MLXRealModelCatalogTests {
    @Test("catalog is unique and includes all registered architecture labels")
    func catalogIsUniqueAndCoversRegisteredArchitectures() throws {
        let models = try MLXRealModelCatalog.load()
        let ids = models.map(\.id)
        let relativePaths = models.map(\.relativePath)
        let architectures = Set(models.map(\.architecture))

        #expect(Set(ids).count == ids.count)
        #expect(Set(relativePaths).count == relativePaths.count)
        #expect(kExpectedCatalogArchitectures.subtracting(architectures).isEmpty)
        #expect(models.contains { $0.tags.contains("latest") })
        #expect(models.contains { $0.tags.contains("smoke") })
    }

    @Test("downloadable catalog entries have Hugging Face repositories")
    func downloadableCatalogEntriesHaveHuggingFaceRepositories() throws {
        let models = try MLXRealModelCatalog.load()
        let downloadable = models.filter(\.isDownloadable)

        #expect(!downloadable.isEmpty)
        #expect(downloadable.allSatisfy { model in
            guard let repository = model.repository else {
                return false
            }
            return repository.split(separator: "/").count == 2
        })
    }

    @Test("known oversized catalog entries declare memory requirements")
    func knownOversizedCatalogEntriesDeclareMemoryRequirements() throws {
        let models = try MLXRealModelCatalog.load()
        let gptOSS = try #require(models.first { $0.id == "gpt-oss" })
        let mistralSmall = try #require(models.first { $0.id == "mistral-small-24b-2501-4bit" })
        let qwen3Next = try #require(models.first { $0.id == "qwen3-next" })
        let qwen35MoE = try #require(models.first { $0.id == "qwen3.5-moe" })
        let gemma3nE2BBF16 = try #require(models.first { $0.id == "gemma-3n-e2b-it-lm-bf16" })
        let teleChat3 = try #require(models.first { $0.id == "telechat3-36b-thinking-4bit" })
        let kimiK25 = try #require(models.first { $0.id == "kimi-k2.5-3bit" })
        let longcatFlash = try #require(models.first { $0.id == "longcat-flash-chat-4bit" })
        let longcatLite = try #require(models.first { $0.id == "longcat-flash-lite-4bit" })

        #expect(gptOSS.minimumMemoryGB == 48)
        #expect(gptOSS.minimumDiskGB == 14)
        #expect(mistralSmall.minimumMemoryGB == 48)
        #expect(mistralSmall.minimumDiskGB == 16)
        #expect(qwen3Next.minimumMemoryGB == 64)
        #expect(qwen3Next.minimumDiskGB == 45)
        #expect(qwen35MoE.minimumMemoryGB == 48)
        #expect(qwen35MoE.minimumDiskGB == 28)
        #expect(gemma3nE2BBF16.minimumMemoryGB == 48)
        #expect(gemma3nE2BBF16.minimumDiskGB == 8)
        #expect(teleChat3.minimumMemoryGB == 48)
        #expect(teleChat3.minimumDiskGB == 24)
        #expect(kimiK25.minimumMemoryGB == 512)
        #expect(kimiK25.minimumDiskGB == 420)
        #expect(longcatFlash.minimumMemoryGB == 384)
        #expect(longcatFlash.minimumDiskGB == 300)
        #expect(longcatLite.minimumMemoryGB == 48)
        #expect(longcatLite.minimumDiskGB == 40)
    }

    @Test("large downloadable catalog entries declare resource requirements")
    func largeDownloadableCatalogEntriesDeclareResourceRequirements() throws {
        let models = try MLXRealModelCatalog.load()
        let largeDownloadable = models.filter { model in
            model.isDownloadable && model.tags.contains("large")
        }

        #expect(!largeDownloadable.isEmpty)
        #expect(largeDownloadable.allSatisfy { ($0.minimumMemoryGB ?? 0) > 0 })
        #expect(largeDownloadable.allSatisfy { ($0.minimumDiskGB ?? 0) > 0 })
    }

    @Test("Klear declares its serialized-run memory guard policy")
    func klearDeclaresSerializedRunMemoryGuardPolicy() throws {
        let models = try MLXRealModelCatalog.load()
        let klear = try #require(models.first { $0.id == "klear-46b-a2.5b-instruct-3bit" })

        #expect(klear.minimumMemoryGB == 32)
        #expect(klear.memoryGuardTier == "off")
    }

    @Test("IQuest Loop Coder declares its oversized guard policy")
    func iQuestLoopCoderDeclaresOversizedGuardPolicy() throws {
        let models = try MLXRealModelCatalog.load()
        let iQuestLoopCoder = try #require(
            models.first { $0.id == "iquest-coder-v1-40b-loop-instruct-4bit" }
        )

        #expect(iQuestLoopCoder.minimumMemoryGB == 48)
        #expect(iQuestLoopCoder.minimumDiskGB == 24)
    }

    @Test("resource gate uses artifact-size fallback when catalog metadata is absent")
    func resourceGateUsesArtifactSizeFallbackWhenCatalogMetadataIsAbsent() {
        let small = Self.fixtureModel(id: "small")
        let large = Self.fixtureModel(id: "large")
        let explicitLarge = Self.fixtureModel(id: "explicit-large", minimumMemoryGB: 48)
        let explicitFit = Self.fixtureModel(id: "explicit-fit", minimumMemoryGB: 24)

        #expect(MLXRealModelEnvironment.canRunWithinHostMemory(
            small,
            estimatedModelLoadBytes: 4 * Self.gib,
            hostMemoryGB: 32
        ))
        #expect(!MLXRealModelEnvironment.canRunWithinHostMemory(
            large,
            estimatedModelLoadBytes: 11 * Self.gib,
            hostMemoryGB: 32
        ))
        #expect(!MLXRealModelEnvironment.canRunWithinHostMemory(
            explicitLarge,
            estimatedModelLoadBytes: 1 * Self.gib,
            hostMemoryGB: 32
        ))
        #expect(MLXRealModelEnvironment.canRunWithinHostMemory(
            explicitFit,
            estimatedModelLoadBytes: 20 * Self.gib,
            hostMemoryGB: 32
        ))
        #expect(MLXRealModelEnvironment.estimatedRuntimeMemoryGB(
            forModelLoadBytes: 11 * Self.gib
        ) == 28)
    }

    @Test("main scope includes representative downloadable architectures")
    func mainScopeIncludesRepresentativeDownloadableArchitectures() throws {
        let models = try MLXRealModelCatalog.load()
        let mainModels = models.filter { $0.tags.contains("main") }
        let architectures = Set(mainModels.map(\.architecture))
        let allMainModelsDownloadable = mainModels.allSatisfy(\.isDownloadable)

        #expect(!mainModels.isEmpty)
        #expect(allMainModelsDownloadable)
        #expect(kExpectedMainCatalogArchitectures.subtracting(architectures).isEmpty)
    }

    @Test("relevant scope includes current downloadable architecture representatives")
    func relevantScopeIncludesCurrentDownloadableArchitectureRepresentatives() throws {
        let models = try MLXRealModelCatalog.load()
        let relevantModels = models.filter { $0.tags.contains("relevant") }
        let architectures = Set(relevantModels.map(\.architecture))
        let allRelevantModelsDownloadable = relevantModels.allSatisfy(\.isDownloadable)

        #expect(!relevantModels.isEmpty)
        #expect(allRelevantModelsDownloadable)
        #expect(kExpectedRelevantCatalogArchitectures.subtracting(architectures).isEmpty)
    }

    private static let gib: Int64 = 1_073_741_824

    private static func fixtureModel(
        id: String,
        minimumMemoryGB: Int? = nil
    ) -> MLXRealModelCatalog.Model {
        MLXRealModelCatalog.Model(
            id: id,
            displayName: id,
            architecture: "fixture",
            repository: "mlx-community/\(id)",
            relativePath: id,
            prompt: "Prompt",
            expectedTokens: [],
            maxTokens: 1,
            minimumMemoryGB: minimumMemoryGB,
            minimumDiskGB: nil,
            memoryGuardTier: nil,
            tags: []
        )
    }
}

private let kExpectedCatalogArchitectures: Set<String> = [
    "Klear",
    "acereason",
    "afm7",
    "afmoe",
    "apertus",
    "baichuan_m1",
    "bailing_moe",
    "bitnet",
    "cohere",
    "cohere2",
    "dbrx",
    "deepseek_v3",
    "ernie4_5",
    "ernie4_5_moe",
    "exaone",
    "exaone4",
    "exaone_moe",
    "falcon_h1",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma3n",
    "gemma4",
    "gemma4_text",
    "glm",
    "glm4",
    "glm4_moe",
    "glm4_moe_lite",
    "gpt2",
    "gpt_bigcode",
    "gpt_oss",
    "gpt_neox",
    "granite",
    "granitemoehybrid",
    "helium",
    "hunyuan",
    "hunyuan_v1_dense",
    "internlm2",
    "iquestloopcoder",
    "jamba",
    "jamba_3b",
    "kimi_k25",
    "lfm2",
    "lfm2_moe",
    "lille-130m",
    "llama",
    "llama4",
    "llama4_text",
    "longcat_flash",
    "longcat_flash_ngram",
    "mamba",
    "mamba2",
    "mellum",
    "mimo",
    "mimo_v2_flash",
    "minicpm",
    "minicpm3",
    "minimax",
    "mistral",
    "mistral3",
    "nanochat",
    "nemotron",
    "nemotron-nas",
    "nemotron_h",
    "olmo",
    "olmo2",
    "olmo3",
    "olmoe",
    "openelm",
    "plamo",
    "plamo2",
    "phi",
    "phi3",
    "phixtral",
    "phimoe",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_text",
    "qwen3_moe",
    "qwen3_next",
    "rwkv7",
    "seed_oss",
    "solar_open",
    "smollm3",
    "stablelm",
    "starcoder2",
    "telechat3",
    "youtu_llm"
]

private let kExpectedMainCatalogArchitectures: Set<String> = [
    "Klear",
    "bitnet",
    "cohere2",
    "dbrx",
    "ernie4_5",
    "ernie4_5_moe",
    "exaone",
    "exaone4",
    "exaone_moe",
    "gemma3",
    "gemma4",
    "glm",
    "granite",
    "gpt2",
    "gpt_bigcode",
    "gpt_neox",
    "helium",
    "hunyuan",
    "hunyuan_v1_dense",
    "jamba",
    "lfm2",
    "lfm2_moe",
    "llama",
    "llama4_text",
    "mamba",
    "mamba2",
    "mellum",
    "minicpm3",
    "mistral",
    "olmo",
    "openelm",
    "plamo2",
    "phi3",
    "phixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "rwkv7",
    "seed_oss",
    "smollm3",
    "stablelm",
    "starcoder2",
    "youtu_llm"
]

private let kExpectedRelevantCatalogArchitectures: Set<String> = [
    "Klear",
    "acereason",
    "apertus",
    "baichuan_m1",
    "bitnet",
    "cohere2",
    "dbrx",
    "deepseek_v3",
    "ernie4_5",
    "ernie4_5_moe",
    "exaone",
    "exaone4",
    "exaone_moe",
    "falcon_h1",
    "gemma3",
    "gemma3n",
    "gemma4",
    "glm",
    "glm4",
    "gpt2",
    "gpt_bigcode",
    "gpt_oss",
    "gpt_neox",
    "granite",
    "helium",
    "hunyuan",
    "hunyuan_v1_dense",
    "jamba",
    "iquestloopcoder",
    "kimi_k25",
    "lfm2",
    "lfm2_moe",
    "llama",
    "llama4_text",
    "mamba",
    "mamba2",
    "mellum",
    "mimo",
    "minicpm3",
    "mistral",
    "olmo",
    "olmo3",
    "openelm",
    "plamo2",
    "phi3",
    "phixtral",
    "qwen2_moe",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_moe",
    "rwkv7",
    "seed_oss",
    "smollm3",
    "stablelm",
    "starcoder2",
    "youtu_llm"
]
