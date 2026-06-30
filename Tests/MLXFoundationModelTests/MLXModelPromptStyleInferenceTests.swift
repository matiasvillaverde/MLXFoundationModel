import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model prompt style inference")
struct MLXModelPromptStyleInferenceTests {
    struct Fixture {
        let id: String?
        let modelType: String
        let template: String
        let style: MLXPromptStyle

        init(
            modelType: String,
            template: String,
            style: MLXPromptStyle,
            id: String?
        ) {
            self.id = id
            self.modelType = modelType
            self.template = template
            self.style = style
        }
    }

    @Test(
        "infers native prompt styles for tool-capable families",
        arguments: fixtures
    )
    func infersNativePromptStylesForToolCapableFamilies(_ fixture: Fixture) {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": fixture.modelType,
                "architectures": ["FixtureForCausalLM"]
            ],
            tokenizerConfig: [
                "chat_template": fixture.template
            ],
            id: fixture.id
        )

        #expect(profile.promptStyle == fixture.style)
    }

    @Test(
        "infers native prompt styles from model ids when templates are missing",
        arguments: idFixtures
    )
    func infersNativePromptStylesFromModelIDs(_ fixture: Fixture) {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": fixture.modelType,
                "architectures": ["GenericForCausalLM"]
            ],
            id: fixture.id
        )

        #expect(profile.promptStyle == fixture.style)
    }

    @Test("prefers explicit template markers over broad config families")
    func prefersExplicitTemplateMarkersOverBroadConfigFamilies() {
        let profile = Self.profile(
            modelType: "gemma",
            template: "<start_function_call>call:weather{}<end_function_call>",
            id: "qwen-misleading-name"
        )

        #expect(profile.promptStyle == .functionGemma)
    }

    @Test("uses architecture families before identifier fallback")
    func usesArchitectureFamiliesBeforeIdentifierFallback() {
        let profile = Self.profile(
            modelType: "unknown",
            architectures: ["MiniMaxM3ForCausalLM"],
            id: "Qwen-misleading-name"
        )

        #expect(profile.promptStyle == .minimaxM3)
    }

    @Test("falls back to ChatML only for templates without native markers")
    func fallsBackToChatMLOnlyForTemplatesWithoutNativeMarkers() {
        let profile = Self.profile(
            modelType: "unknown",
            template: "{{ messages }}"
        )

        #expect(profile.promptStyle == .chatML)
    }

    @Test("infers Mixtral as Mistral family before generic ChatML fallback")
    func infersMixtralAsMistralFamilyBeforeGenericChatMLFallback() {
        let profile = Self.profile(
            modelType: "mixtral",
            architectures: ["MixtralForCausalLM"],
            template: "<|im_start|>{{ role }}\n{{ content }}<|im_end|>"
        )

        #expect(profile.promptStyle == .mistralToolCall)
    }

    private static let fixtures = [
        Fixture(
            modelType: "apertus",
            template: "<|user_start|>hello<|user_end|><|assistant_start|>",
            style: .apertus,
            id: nil
        ),
        Fixture(
            modelType: "command-r",
            template: "<|START_ACTION|>{}</|END_ACTION|>",
            style: .cohereAction,
            id: nil
        ),
        Fixture(
            modelType: "kimi_k2",
            template: "<|tool_calls_section_begin|>",
            style: .kimiK2,
            id: nil
        ),
        Fixture(
            modelType: "longcat_flash",
            template: "<longcat_tool_call></longcat_tool_call>",
            style: .longCat,
            id: nil
        ),
        Fixture(
            modelType: "mistral",
            template: "[TOOL_CALLS]weather[ARGS]{}",
            style: .mistralToolCall,
            id: nil
        ),
        Fixture(
            modelType: "gemma4",
            template: "<|tool_call>call:weather{}<tool_call|>",
            style: .gemma,
            id: nil
        ),
        Fixture(
            modelType: "deepseek_v4",
            template: "<｜DSML｜tool_calls></｜DSML｜tool_calls>",
            style: .deepSeekDSML,
            id: nil
        ),
        Fixture(
            modelType: "glm4_moe",
            template: "<arg_key>city</arg_key>",
            style: .glmXML,
            id: nil
        ),
        Fixture(
            modelType: "qwen3_5",
            template: "<tool_call><function=weather></function></tool_call>",
            style: .qwenXML,
            id: nil
        ),
        Fixture(
            modelType: "minimax",
            template: #"<minimax:tool_call><invoke name="weather">"#,
            style: .minimaxXML,
            id: nil
        ),
        Fixture(
            modelType: "minimax_m3",
            template: "]<]minimax[>[<tool_call>",
            style: .minimaxM3,
            id: nil
        )
    ]

    private static let idFixtures = [
        Fixture(modelType: "unknown", template: "", style: .apertus, id: "Apertus-8B-Instruct"),
        Fixture(modelType: "unknown", template: "", style: .qwenXML, id: "Qwen3-0.6B-4bit"),
        Fixture(modelType: "unknown", template: "", style: .glmXML, id: "GLM-4.7-9B-4bit"),
        Fixture(modelType: "unknown", template: "", style: .minimaxM3, id: "MiniMax-M3-4bit"),
        Fixture(modelType: "unknown", template: "", style: .kimiK2, id: "Kimi-K2-Instruct-4bit"),
        Fixture(modelType: "unknown", template: "", style: .longCat, id: "LongCat-Flash-Chat"),
        Fixture(modelType: "unknown", template: "", style: .functionGemma, id: "Function-Gemma-2B"),
        Fixture(modelType: "unknown", template: "", style: .cohereAction, id: "Command-R-08-2024")
    ]

    private static func profile(
        modelType: String,
        architectures: [String] = ["GenericForCausalLM"],
        template: String = "",
        id: String? = nil
    ) -> MLXModelProfile {
        MLXModelProfile.make(
            config: [
                "model_type": modelType,
                "architectures": architectures
            ],
            tokenizerConfig: [
                "chat_template": template
            ],
            id: id
        )
    }
}
