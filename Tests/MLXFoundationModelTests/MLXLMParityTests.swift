@testable import MLXLocalModels
import Testing

@Suite("MLX-LM architecture parity")
struct MLXLMParityTests {
    @Test("known Foundation Models-compatible text gaps stay explicit")
    func knownTextGapsStayExplicit() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        let deferredTypes = Self.mlxLMTextModelTypes.subtracting(registeredTypes)

        #expect(deferredTypes == Self.expectedDeferredTextModelTypes)
        #expect(Self.expectedDeferredTextModelTypes.isDisjoint(with: registeredTypes))
    }

    @Test("non-text model families stay outside the Foundation Models target")
    func nonTextFamiliesStayOutsideFoundationModelsTarget() {
        #expect(Self.nonTextModelTypes.isDisjoint(with: Self.mlxLMTextModelTypes))
    }

    // Snapshot from ml-explore/mlx-lm model modules on 2026-07-01.
    // This list intentionally tracks text-generation families only. VLM, OCR,
    // embedding, reranker, and helper modules are not part of the Foundation
    // Models-compatible text parity target.
    private static let mlxLMTextModelTypes: Set<String> = [
        "afm7",
        "afmoe",
        "apertus",
        "baichuan_m1",
        "bailing_moe",
        "bailing_moe_linear",
        "bitnet",
        "cohere",
        "cohere2",
        "dbrx",
        "deepseek",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
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
        "glm_moe_dsa",
        "gpt2",
        "gpt_bigcode",
        "gpt_neox",
        "gpt_oss",
        "granite",
        "granitemoe",
        "granitemoehybrid",
        "helium",
        "hunyuan",
        "hunyuan_v1_dense",
        "internlm2",
        "iquestloopcoder",
        "jamba",
        "kimi_k25",
        "kimi_linear",
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
        "ministral3",
        "mistral3",
        "mixtral",
        "nanochat",
        "nemotron",
        "nemotron-nas",
        "nemotron_h",
        "olmo",
        "olmo2",
        "olmo3",
        "olmoe",
        "openelm",
        "phi",
        "phi3",
        "phi3small",
        "phimoe",
        "phixtral",
        "plamo",
        "plamo2",
        "qwen",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_moe",
        "qwen3_next",
        "recurrent_gemma",
        "rwkv7",
        "seed_oss",
        "smollm3",
        "solar_open",
        "stablelm",
        "starcoder2",
        "step3p5",
        "telechat3",
        "youtu_llm"
    ]

    private static let expectedDeferredTextModelTypes: Set<String> = [
        "afm7",
        "bailing_moe_linear",
        "kimi_linear",
        "recurrent_gemma",
        "step3p5"
    ]

    private static let nonTextModelTypes: Set<String> = [
        "clip",
        "deepseek_ocr",
        "dots1",
        "lfm2-vl",
        "kimi_vl",
        "paddleocr",
        "pixtral",
        "qwen2_vl",
        "qwen3_vl",
        "qwen3_vl_moe"
    ]
}
