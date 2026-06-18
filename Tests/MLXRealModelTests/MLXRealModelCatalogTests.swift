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
        #expect(Self.expectedArchitectures.subtracting(architectures).isEmpty)
        #expect(models.contains { $0.tags.contains("latest") })
        #expect(models.contains { $0.tags.contains("smoke") })
    }

    @Test("downloadable catalog entries have repositories")
    func downloadableCatalogEntriesHaveRepositories() throws {
        let models = try MLXRealModelCatalog.load()
        let downloadable = models.filter(\.isDownloadable)

        #expect(!downloadable.isEmpty)
        #expect(downloadable.allSatisfy { $0.repository?.hasPrefix("mlx-community/") == true })
    }

    @Test("main scope includes representative downloadable architectures")
    func mainScopeIncludesRepresentativeDownloadableArchitectures() throws {
        let models = try MLXRealModelCatalog.load()
        let mainModels = models.filter { $0.tags.contains("main") }
        let architectures = Set(mainModels.map(\.architecture))
        let allMainModelsDownloadable = mainModels.allSatisfy(\.isDownloadable)

        #expect(!mainModels.isEmpty)
        #expect(allMainModelsDownloadable)
        #expect(Self.expectedMainArchitectures.subtracting(architectures).isEmpty)
    }

    private static let expectedArchitectures: Set<String> = [
        "acereason",
        "afmoe",
        "apertus",
        "baichuan_m1",
        "bailing_moe",
        "bitnet",
        "cohere",
        "deepseek_v3",
        "ernie4_5",
        "exaone4",
        "falcon_h1",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma3n",
        "gemma4",
        "gemma4_text",
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "gpt_oss",
        "granite",
        "granitemoehybrid",
        "internlm2",
        "jamba_3b",
        "lfm2",
        "lfm2_moe",
        "lille-130m",
        "llama",
        "mimo",
        "mimo_v2_flash",
        "minicpm",
        "minimax",
        "mistral",
        "mistral3",
        "nanochat",
        "nemotron_h",
        "olmo2",
        "olmo3",
        "olmoe",
        "openelm",
        "phi",
        "phi3",
        "phimoe",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_text",
        "qwen3_moe",
        "qwen3_next",
        "smollm3",
        "starcoder2"
    ]

    private static let expectedMainArchitectures: Set<String> = [
        "bitnet",
        "ernie4_5",
        "exaone4",
        "gemma3",
        "gemma4",
        "granite",
        "lfm2",
        "llama",
        "mistral",
        "openelm",
        "phi3",
        "qwen2",
        "qwen3",
        "smollm3",
        "starcoder2"
    ]
}
