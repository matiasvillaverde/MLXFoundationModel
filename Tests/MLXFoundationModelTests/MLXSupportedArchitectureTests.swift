@testable import MLXLocalModels
import Testing

@Suite("MLX supported architecture registry")
struct MLXSupportedArchitectureTests {
    @Test("registers every copied MLX model family")
    func registersEveryCopiedMLXModelFamily() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(Self.expectedTypes.subtracting(registeredTypes).isEmpty)
    }

    private static let expectedTypes: Set<String> = [
        "acereason",
        "afmoe",
        "apertus",
        "baichuan_m1",
        "bailing_moe",
        "bitnet",
        "cohere",
        "deepseek_v3",
        "deepseek_v32",
        "ernie4_5",
        "exaone4",
        "falcon_h1",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma3n",
        "gemma4",
        "gemma4_assistant",
        "gemma4_text",
        "gemma4_unified_assistant",
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
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
        "ministral3",
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
}
