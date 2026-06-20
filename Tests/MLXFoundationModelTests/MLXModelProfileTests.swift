import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model profiles")
struct MLXModelProfileTests {
    @Test("infers Qwen profile metadata from config and chat template")
    func infersQwenProfileMetadata() throws {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "max_position_embeddings": 32_768,
                "vocab_size": 151_936,
                "quantization": ["bits": 4]
            ],
            tokenizerConfig: [
                "chat_template": "<tool_call><function=weather></function></tool_call>"
            ],
            id: "qwen3-0.6b-4bit"
        )

        #expect(profile.id == "qwen3-0.6b-4bit")
        #expect(profile.modelType == "qwen3")
        #expect(profile.architectures == ["Qwen3ForCausalLM"])
        #expect(profile.contextLength == 32_768)
        #expect(profile.vocabularySize == 151_936)
        #expect(profile.quantizationBits == 4)
        #expect(profile.optimizationProfile.quantization?.bits == 4)
        #expect(profile.hasNativeChatTemplate)
        #expect(profile.promptStyle == .qwenXML)
        #expect(profile.capabilities.toolCalling)
        #expect(profile.capabilities.structuredOutput)
        #expect(profile.capabilities.reasoning)
        #expect(!profile.capabilities.vision)
    }

    @Test("surfaces MoE and vision metadata in public capabilities")
    func surfacesMoEAndVisionMetadata() {
        let profile = MLXModelProfile.make(config: [
            "model_type": "glm4_moe",
            "architectures": ["Glm4MoeForCausalLM"],
            "num_experts": 128,
            "vision_config": ["image_size": 448],
            "chat_template": "<arg_key>city</arg_key><arg_value>Berlin</arg_value>"
        ])

        #expect(profile.isMixtureOfExperts)
        #expect(profile.hasVisionConfig)
        #expect(profile.promptStyle == .glmXML)
        #expect(profile.capabilities.reasoning)
        #expect(profile.capabilities.vision)
    }

    @Test("loads profiles from a model directory")
    func loadsProfilesFromModelDirectory() throws {
        let directory = try Self.makeTemporaryModelDirectory(
            config: [
                "model_type": "minimax",
                "architectures": ["MiniMaxTextModel"],
                "seq_length": 65_536,
                "quantization_config": ["nbits": 4]
            ],
            tokenizerConfig: [
                "chat_template": "<minimax:tool_call><invoke name=\"weather\" /></minimax:tool_call>"
            ]
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let model = try MLXModel.profiled(id: "minimax-fixture", location: directory)
        let profile = try #require(model.profile)

        #expect(model.promptStyle == .minimaxXML)
        #expect(model.capabilities.structuredOutput)
        #expect(profile.contextLength == 65_536)
        #expect(profile.quantizationBits == 4)
        #expect(profile.optimizationProfile.quantization?.bits == 4)
    }

    @Test("language model convenience initializer loads profile defaults")
    func languageModelConvenienceInitializerLoadsProfileDefaults() throws {
        let directory = try Self.makeTemporaryModelDirectory(
            config: [
                "model_type": "glm4_moe",
                "architectures": ["Glm4MoeForCausalLM"],
                "num_experts": 32
            ],
            tokenizerConfig: [
                "chat_template": "<arg_key>city</arg_key><arg_value>Berlin</arg_value>"
            ]
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let languageModel = try MLXLanguageModel(
            id: "glm-fixture",
            location: directory,
            maximumResponseTokens: 128
        )
        let profile = try #require(languageModel.profile)

        #expect(languageModel.model.promptStyle == .glmXML)
        #expect(languageModel.model.capabilities.toolCalling)
        #expect(languageModel.maximumResponseTokens == 128)
        #expect(profile.isMixtureOfExperts)
    }

    @Test("infers MiniMax M3 native prompt style")
    func infersMiniMaxM3NativePromptStyle() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "minimax_m3",
                "architectures": ["MiniMaxM3ForCausalLM"],
                "chat_template": "]<]minimax[>[<tool_call>]<]minimax[>[</tool_call>"
            ]
        )

        #expect(profile.promptStyle == .minimaxM3)
    }

    @Test("infers Harmony prompt style for gpt oss models")
    func infersHarmonyPromptStyleForGptOssModels() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "gpt_oss",
                "architectures": ["GptOssForCausalLM"],
                "chat_template": "<|start|>assistant<|channel|>final<|message|>"
            ],
            id: "gpt-oss-20b"
        )

        #expect(profile.promptStyle == .harmony)
        #expect(profile.capabilities.reasoning)
    }

    @Test("infers legacy FunctionGemma prompt style before generic Gemma")
    func infersLegacyFunctionGemmaPromptStyle() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "gemma",
                "architectures": ["GemmaForCausalLM"]
            ],
            tokenizerConfig: [
                "chat_template": "<start_function_call>call:name{}<end_function_call>"
            ],
            id: "function-gemma-fixture"
        )

        #expect(profile.promptStyle == .functionGemma)
    }

    @Test("explicit model settings override profile defaults")
    func explicitModelSettingsOverrideProfileDefaults() {
        let profile = MLXModelProfile(
            promptStyle: .qwenXML,
            capabilities: MLXModelCapabilities(toolCalling: true, structuredOutput: true)
        )
        let model = MLXModel(
            id: "manual",
            location: URL(fileURLWithPath: "/tmp/manual"),
            promptStyle: .plain,
            capabilities: MLXModelCapabilities(toolCalling: false),
            profile: profile
        )

        #expect(model.promptStyle == .plain)
        #expect(!model.capabilities.toolCalling)
        #expect(model.profile == profile)
    }

    @Test("infers oMLX optimization metadata from model config")
    func infersOMLXOptimizationMetadataFromModelConfig() {
        let optimization = Self.deepSeekOptimizationProfile

        Self.expectDeepSeekOptimizationFlags(optimization)
        Self.expectDeepSeekOptimizationFeatureSets(optimization)
    }

    private static var deepSeekOptimizationProfile: MLXModelOptimizationProfile {
        MLXModelProfileFactory.make(
            config: [
                "model_type": "deepseek_v4",
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_nextn_predict_layers": 1,
                "num_experts": 256,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "quantization_config": [
                    "bits": 8,
                    "group_size": 32,
                    "quant_method": "mxfp8"
                ]
            ],
            tokenizerConfig: [:],
            id: "DeepSeek-V4-Flash-oQ3.5e",
            hasNativeMTPWeightTensors: true
        )
        .optimizationProfile
    }

    private static func expectDeepSeekOptimizationFlags(
        _ optimization: MLXModelOptimizationProfile
    ) {
        #expect(optimization.quantization?.bits == 8)
        #expect(optimization.quantization?.groupSize == 32)
        #expect(optimization.quantization?.method == "mxfp8")
        #expect(optimization.isOQQuantized)
        #expect(optimization.oQLevel == "oQ3.5e")
        #expect(optimization.requiresFP8ScaleDequantization)
        #expect(optimization.hasNativeMTPWeights)
        #expect(optimization.supportsNativeMTP)
        #expect(!optimization.nativeMTPRuntimeSupported)
        #expect(!optimization.supportsVLMMTP)
        #expect(optimization.supportsSpeculativePrefill)
        #expect(!optimization.supportsDFlash)
        #expect(optimization.supportsIndexCache)
        #expect(optimization.supportsTurboQuantKV)
    }

    private static func expectDeepSeekOptimizationFeatureSets(
        _ optimization: MLXModelOptimizationProfile
    ) {
        #expect(optimization.detectedFeatures == [
            .fp8ScaleDequantization,
            .indexCache,
            .nativeMTP,
            .oQQuantization,
            .speculativePrefill,
            .turboQuantKV
        ])
        #expect(optimization.implementedFeatures == [
            .fp8ScaleDequantization,
            .indexCache,
            .turboQuantKV
        ])
        #expect(optimization.pendingRuntimeFeatures == [
            .nativeMTP,
            .oQQuantization,
            .speculativePrefill
        ])
    }

    private static func makeTemporaryModelDirectory(
        config: [String: Any],
        tokenizerConfig: [String: Any]
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXModelProfileTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try writeJSON(config, to: directory.appendingPathComponent("config.json"))
        try writeJSON(tokenizerConfig, to: directory.appendingPathComponent("tokenizer_config.json"))
        return directory
    }

    private static func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        try data.write(to: url)
    }
}
