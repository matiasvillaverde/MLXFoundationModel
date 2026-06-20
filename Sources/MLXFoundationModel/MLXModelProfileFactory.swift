import Foundation

enum MLXModelProfileFactory {
    private struct ProfileTraits {
        let isMixtureOfExperts: Bool
        let hasVisionConfig: Bool
        let isVisionModel: Bool
        let runtimeKind: MLXModelRuntimeKind
    }

    private struct ProfileInput {
        let config: [String: Any]
        let tokenizerConfig: [String: Any]
        let id: String?
        let standaloneChatTemplate: String?
        let hasNativeMTPWeightTensors: Bool
        let hasFP8ScaleSidecars: Bool
    }

    private struct ProfileBuildContext {
        let id: String?
        let config: [String: Any]
        let modelType: String?
        let architectures: [String]
        let template: String?
        let traits: ProfileTraits
        let defaultReasoning: MLXModelReasoningDefault?
        let hasNativeMTPWeightTensors: Bool
        let hasFP8ScaleSidecars: Bool
    }

    static func load(
        from modelDirectory: URL,
        id: String?
    ) throws -> MLXModelProfile {
        let config = try loadJSON(
            at: modelDirectory.appendingPathComponent("config.json")
        )
        let tokenizerConfig = try loadOptionalJSON(
            at: modelDirectory.appendingPathComponent("tokenizer_config.json")
        )
        let standaloneChatTemplate = try loadOptionalText(
            at: modelDirectory.appendingPathComponent("chat_template.jinja")
        )
        let evidence = MLXModelWeightArtifactScanner.evidence(in: modelDirectory)
        return make(
            config: config,
            tokenizerConfig: tokenizerConfig,
            id: id,
            standaloneChatTemplate: standaloneChatTemplate,
            hasNativeMTPWeightTensors: evidence.hasMTPWeightTensors,
            hasFP8ScaleSidecars: evidence.hasFP8ScaleSidecars
        )
    }

    static func make(
        config: [String: Any],
        tokenizerConfig: [String: Any],
        id: String?,
        standaloneChatTemplate: String? = nil,
        hasNativeMTPWeightTensors: Bool = false,
        hasFP8ScaleSidecars: Bool = false
    ) -> MLXModelProfile {
        profile(buildContext(
            .init(
                config: config,
                tokenizerConfig: tokenizerConfig,
                id: id,
                standaloneChatTemplate: standaloneChatTemplate,
                hasNativeMTPWeightTensors: hasNativeMTPWeightTensors,
                hasFP8ScaleSidecars: hasFP8ScaleSidecars
            )
        ))
    }

    private static func buildContext(_ input: ProfileInput) -> ProfileBuildContext {
        let modelType = string(input.config, keys: ["model_type"])
        let architectures = stringArray(input.config, keys: ["architectures"])
        let template = chatTemplate(
            config: input.config,
            tokenizerConfig: input.tokenizerConfig,
            standaloneTemplate: input.standaloneChatTemplate
        )
        return ProfileBuildContext(
            id: input.id,
            config: input.config,
            modelType: modelType,
            architectures: architectures,
            template: template,
            traits: profileTraits(modelType: modelType, architectures: architectures, config: input.config),
            defaultReasoning: defaultReasoning(in: template),
            hasNativeMTPWeightTensors: input.hasNativeMTPWeightTensors,
            hasFP8ScaleSidecars: input.hasFP8ScaleSidecars
        )
    }

    private static func profile(_ context: ProfileBuildContext) -> MLXModelProfile {
        let promptStyle = inferredStyle(
            context.id,
            context.modelType,
            context.architectures,
            context.template
        )
        let capabilities = inferCapabilities(
            id: context.id,
            modelType: context.modelType,
            architectures: context.architectures,
            isVisionModel: context.traits.isVisionModel,
            defaultReasoning: context.defaultReasoning
        )
        return MLXModelProfile(
            id: context.id,
            modelType: context.modelType,
            architectures: context.architectures,
            contextLength: contextLength(context.config),
            vocabularySize: int(context.config, keys: ["vocab_size", "padded_vocab_size"]),
            quantizationBits: quantizationBits(context.config),
            isMixtureOfExperts: context.traits.isMixtureOfExperts,
            hasVisionConfig: context.traits.hasVisionConfig,
            hasNativeChatTemplate: context.template?.isEmpty == false,
            defaultReasoning: context.defaultReasoning,
            promptStyle: promptStyle,
            runtimeKind: context.traits.runtimeKind,
            capabilities: capabilities,
            optimization: opt(context)
        )
    }

    private static func profileTraits(
        modelType: String?,
        architectures: [String],
        config: [String: Any]
    ) -> ProfileTraits {
        let hasVisionConfig = MLXVisionModelClassifier.hasVisionEvidence(in: config)
        let isVisionModel = MLXVisionModelClassifier.isVisionModel(
            modelType: modelType,
            architectures: architectures,
            hasVisionEvidence: hasVisionConfig
        )
        return ProfileTraits(
            isMixtureOfExperts: isMoE(modelType: modelType, config: config),
            hasVisionConfig: hasVisionConfig,
            isVisionModel: isVisionModel,
            runtimeKind: MLXVisionModelClassifier.runtimeKind(
                modelType: modelType,
                architectures: architectures,
                isVisionModel: isVisionModel
            )
        )
    }

    private static func inferPromptStyle(
        id: String?,
        modelType: String?,
        architectures: [String],
        chatTemplate: String?
    ) -> MLXPromptStyle {
        MLXPromptStyleInferenceEngine.promptStyle(
            for: .init(
                id: id,
                modelType: modelType,
                architectures: architectures,
                chatTemplate: chatTemplate
            )
        )
    }

    private static func inferredStyle(
        _ id: String?,
        _ modelType: String?,
        _ architectures: [String],
        _ template: String?
    ) -> MLXPromptStyle {
        inferPromptStyle(id: id, modelType: modelType, architectures: architectures, chatTemplate: template)
    }

    private static func inferCapabilities(
        id: String?,
        modelType: String?,
        architectures: [String],
        isVisionModel: Bool,
        defaultReasoning: MLXModelReasoningDefault?
    ) -> MLXModelCapabilities {
        MLXModelCapabilities(
            toolCalling: true,
            structuredOutput: true,
            vision: isVisionModel,
            reasoning: defaultReasoning != nil ||
                infersReasoning(id: id, modelType: modelType, architectures: architectures)
        )
    }

    private static func infersReasoning(
        id: String?,
        modelType: String?,
        architectures: [String]
    ) -> Bool {
        let text = searchableText(modelType: modelType, architectures: architectures, extra: [id])
        return containsAny(text, [
            "deepseek",
            "reason",
            "thinking",
            "gpt_oss",
            "gpt-oss",
            "gptoss",
            "qwen3",
            "glm4",
            "lfm2"
        ])
    }

    private static func searchableText(
        modelType: String?,
        architectures: [String],
        extra: [String?]
    ) -> String {
        ([modelType] + architectures.map(Optional.some) + extra)
            .compactMap(\.self)
            .joined(separator: "\n")
            .lowercased()
    }

    private static func isMoE(modelType: String?, config: [String: Any]) -> Bool {
        let type = modelType?.lowercased() ?? ""
        return type.contains("moe") ||
            int(config, keys: ["num_experts", "n_routed_experts", "num_local_experts"]) != nil
    }

    private static func defaultReasoning(in template: String?) -> MLXModelReasoningDefault? {
        guard let template else {
            return nil
        }
        let normalized = template.lowercased()
        guard normalized.contains("enable_thinking") else {
            return nil
        }
        if normalized.contains("enable_thinking is false") {
            return .enabled
        }
        if normalized.contains("default(false)") || normalized.contains("enable_thinking)") {
            return .disabled
        }
        return nil
    }

    private static func opt(_ context: ProfileBuildContext) -> MLXModelOptimizationProfile {
        MLXModelOptimizationProfileFactory.make(
            input: .init(
                id: context.id,
                modelType: context.modelType,
                architectures: context.architectures,
                config: context.config,
                isMixtureOfExperts: context.traits.isMixtureOfExperts,
                isVisionModel: context.traits.isVisionModel,
                requiresVLMRuntime: context.traits.runtimeKind == .vlm,
                hasNativeMTPWeightTensors: context.hasNativeMTPWeightTensors,
                hasFP8ScaleSidecars: context.hasFP8ScaleSidecars
            )
        )
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }
}
