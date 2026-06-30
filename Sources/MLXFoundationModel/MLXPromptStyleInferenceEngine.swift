enum MLXPromptStyleInferenceEngine {
    struct Input: Sendable {
        let id: String?
        let modelType: String?
        let architectures: [String]
        let chatTemplate: String?
    }

    private protocol Strategy: Sendable {
        func promptStyle(for input: Input) -> MLXPromptStyle?
    }

    private struct TemplateMarkerStrategy: Strategy {
        func promptStyle(for input: Input) -> MLXPromptStyle? {
            MLXPromptStyleInferenceEngine.orderedRules.first { rule in
                rule.matchesTemplate(input.chatTemplate)
            }?.style
        }
    }

    private struct ModelTypeStrategy: Strategy {
        func promptStyle(for input: Input) -> MLXPromptStyle? {
            MLXPromptStyleInferenceEngine.orderedRules.first { rule in
                rule.matchesModelType(input.modelType)
            }?.style
        }
    }

    private struct ArchitectureStrategy: Strategy {
        func promptStyle(for input: Input) -> MLXPromptStyle? {
            MLXPromptStyleInferenceEngine.orderedRules.first { rule in
                rule.matchesArchitectures(input.architectures)
            }?.style
        }
    }

    private struct IdentifierStrategy: Strategy {
        func promptStyle(for input: Input) -> MLXPromptStyle? {
            MLXPromptStyleInferenceEngine.orderedRules.first { rule in
                rule.matchesIdentifier(input.id)
            }?.style
        }
    }

    private struct Rule: Sendable {
        let style: MLXPromptStyle
        let templateMarkers: [String]
        let modelFamilies: [String]

        init(
            style: MLXPromptStyle,
            templateMarkers: [String],
            modelFamilies: [String]
        ) {
            self.style = style
            self.templateMarkers = templateMarkers
            self.modelFamilies = modelFamilies
        }

        func matchesTemplate(_ template: String?) -> Bool {
            Text(template).containsLiteralAny(templateMarkers)
        }

        func matchesModelType(_ modelType: String?) -> Bool {
            Text(modelType).containsAny(modelFamilies)
        }

        func matchesArchitectures(_ architectures: [String]) -> Bool {
            architectures.contains { architecture in
                Text(architecture).containsAny(modelFamilies)
            }
        }

        func matchesIdentifier(_ id: String?) -> Bool {
            Text(id).containsAny(modelFamilies)
        }
    }

    private struct Text {
        private let lowercased: String
        private let compacted: String

        init(_ value: String?) {
            lowercased = value?.lowercased() ?? ""
            compacted = Self.compact(lowercased)
        }

        func containsAny(_ candidates: [String]) -> Bool {
            candidates.contains(where: contains)
        }

        func containsLiteralAny(_ candidates: [String]) -> Bool {
            candidates.contains { candidate in
                lowercased.contains(candidate.lowercased())
            }
        }

        private func contains(_ candidate: String) -> Bool {
            lowercased.contains(candidate.lowercased()) ||
                compacted.contains(Self.compact(candidate))
        }

        private static func compact(_ value: String) -> String {
            value.lowercased().filter { character in
                character.isLetter || character.isNumber
            }
        }
    }

    private static let strategies: [any Strategy] = [
        TemplateMarkerStrategy(),
        ModelTypeStrategy(),
        ArchitectureStrategy(),
        IdentifierStrategy()
    ]

    private static let orderedRules: [Rule] = [
        .init(
            style: .apertus,
            templateMarkers: ["<|assistant_start|>", "<|user_start|>"],
            modelFamilies: ["apertus"]
        ),
        .init(
            style: .harmony,
            templateMarkers: ["<|channel|>", "gpt_oss"],
            modelFamilies: ["gpt_oss", "gpt-oss", "gptoss", "harmony"]
        ),
        .init(
            style: .deepSeekDSML,
            templateMarkers: ["<｜dsml｜"],
            modelFamilies: ["deepseek_v4", "deepseek-v4", "deepseekv4"]
        ),
        .init(
            style: .cohereAction,
            templateMarkers: ["<|start_action|>"],
            modelFamilies: ["command-r", "command_r", "commandr", "cohere"]
        ),
        .init(
            style: .kimiK2,
            templateMarkers: ["<|tool_calls_section_begin|>"],
            modelFamilies: ["kimi_k2", "kimi-k2", "kimik2"]
        ),
        .init(
            style: .longCat,
            templateMarkers: ["<longcat_tool_call>"],
            modelFamilies: ["longcat", "longcat_flash", "longcat-flash"]
        ),
        .init(
            style: .minimaxM3,
            templateMarkers: [MLXToolPromptDialect.miniMaxNamespaceToken],
            modelFamilies: ["minimax_m3", "minimax-m3", "minimaxm3"]
        ),
        .init(
            style: .minimaxXML,
            templateMarkers: ["<minimax:tool_call>"],
            modelFamilies: ["minimax"]
        ),
        .init(
            style: .mistralToolCall,
            templateMarkers: ["[tool_calls]"],
            modelFamilies: ["mistral", "mixtral"]
        ),
        .init(
            style: .glmXML,
            templateMarkers: ["<arg_key>"],
            modelFamilies: ["glm_moe_dsa", "glm4_moe_lite", "glm4_moe", "glm4", "glm"]
        ),
        .init(
            style: .functionGemma,
            templateMarkers: ["<start_function_call>"],
            modelFamilies: ["function_gemma", "function-gemma", "functiongemma"]
        ),
        .init(
            style: .gemma,
            templateMarkers: ["<|tool_call>"],
            modelFamilies: ["gemma4", "gemma3", "gemma"]
        ),
        .init(
            style: .qwenXML,
            templateMarkers: ["<tool_call><function", "<function="],
            modelFamilies: [
                "qwen3_5_text",
                "qwen3_5_moe",
                "qwen3_next",
                "qwen3_moe",
                "qwen3_5",
                "qwen3",
                "qwen2_5",
                "qwen2",
                "qwen"
            ]
        )
    ]

    static func promptStyle(for input: Input) -> MLXPromptStyle {
        for strategy in strategies {
            if let style = strategy.promptStyle(for: input) {
                return style
            }
        }
        return input.chatTemplate?.isEmpty == false ? .chatML : .plain
    }
}
