import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model profile reasoning defaults")
struct MLXModelProfileReasoningTests {
    @Test("detects chat-template thinking defaults")
    func detectsChatTemplateThinkingDefaults() {
        let qwen = Self.profile(
            template: "{% if enable_thinking is false %}<think disabled>{% endif %}",
            id: "template-default-thinking-fixture"
        )
        let gemma = Self.profile(
            template: "{{ enable_thinking | default(false) }}",
            id: "template-opt-in-thinking-fixture"
        )

        #expect(qwen.defaultReasoning == .enabled)
        #expect(qwen.usesReasoningByDefault)
        #expect(qwen.capabilities.reasoning)
        #expect(gemma.defaultReasoning == .disabled)
        #expect(!gemma.usesReasoningByDefault)
        #expect(gemma.hasReasoningToggle)
        #expect(gemma.capabilities.reasoning)
    }

    @Test("loads standalone chat template before tokenizer config")
    func loadsStandaloneChatTemplateBeforeTokenizerConfig() throws {
        let directory = try Self.makeTemporaryModelDirectory(
            config: [
                "model_type": "unknown",
                "architectures": ["GenericForCausalLM"]
            ],
            tokenizerConfig: [
                "chat_template": "{{ messages }}"
            ],
            chatTemplate: """
            {% if enable_thinking is false %}{% endif %}
            <tool_call><function=weather></function></tool_call>
            """
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let profile = try MLXModelProfile.load(from: directory, id: "standalone-template-fixture")

        #expect(profile.hasNativeChatTemplate)
        #expect(profile.promptStyle == .qwenXML)
        #expect(profile.defaultReasoning == .enabled)
        #expect(profile.usesReasoningByDefault)
        #expect(profile.capabilities.reasoning)
    }

    private static func profile(template: String, id: String) -> MLXModelProfile {
        MLXModelProfile.make(
            config: [
                "model_type": "unknown",
                "architectures": ["GenericForCausalLM"]
            ],
            tokenizerConfig: [
                "chat_template": template
            ],
            id: id
        )
    }

    private static func makeTemporaryModelDirectory(
        config: [String: Any],
        tokenizerConfig: [String: Any],
        chatTemplate: String
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXModelProfileReasoningTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try writeJSON(config, to: directory.appendingPathComponent("config.json"))
        try writeJSON(tokenizerConfig, to: directory.appendingPathComponent("tokenizer_config.json"))
        try chatTemplate.write(
            to: directory.appendingPathComponent("chat_template.jinja"),
            atomically: true,
            encoding: .utf8
        )
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
