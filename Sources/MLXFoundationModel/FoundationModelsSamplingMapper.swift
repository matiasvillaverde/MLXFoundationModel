#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsSamplingMapper {
    private struct BaseSamplingValues {
        var temperature: Float
        var topP: Float
        var topK: Int?
        var seed: Int?
    }

    static func sampling(
        from request: LanguageModelExecutorGenerationRequest,
        toolDefinitions: [Transcript.ToolDefinition],
        promptStyle: MLXPromptStyle,
        fallback: SamplingParameters
    ) -> SamplingParameters {
        let sampling = baseSampling(
            from: request.generationOptions,
            contextOptions: request.contextOptions,
            promptStyle: promptStyle,
            fallback: fallback
        )
        guard let grammar = grammarConfiguration(
            schema: request.schema,
            toolDefinitions: toolDefinitions,
            promptStyle: promptStyle,
            options: request.generationOptions
        ) else {
            return sampling
        }
        return applyingGrammar(grammar, to: sampling)
    }

    static func requiresToolCall(_ options: GenerationOptions) -> Bool {
        switch options.toolCallingMode?.kind {
        case .required:
            true

        case .allowed, .disallowed, nil:
            false

        @unknown default:
            false
        }
    }

    private static func applyingGrammar(
        _ grammar: GrammarSamplingConfiguration,
        to sampling: SamplingParameters
    ) -> SamplingParameters {
        SamplingParameters(
            temperature: sampling.temperature,
            topP: sampling.topP,
            topK: sampling.topK,
            repetitionPenalty: sampling.repetitionPenalty,
            frequencyPenalty: sampling.frequencyPenalty,
            presencePenalty: sampling.presencePenalty,
            repetitionPenaltyRange: sampling.repetitionPenaltyRange,
            seed: sampling.seed,
            stopSequences: sampling.stopSequences,
            advanced: advancedParameters(from: sampling, grammar: grammar)
        )
    }

    private static func advancedParameters(
        from sampling: SamplingParameters,
        grammar: GrammarSamplingConfiguration
    ) -> AdvancedSamplingParameters {
        AdvancedSamplingParameters(
            minP: sampling.advanced.minP,
            typicalP: sampling.advanced.typicalP,
            topNSigma: sampling.advanced.topNSigma,
            grammar: grammar,
            mirostat: sampling.advanced.mirostat,
            xtc: sampling.advanced.xtc,
            dry: sampling.advanced.dry,
            adaptiveP: sampling.advanced.adaptiveP,
            reasoningBudget: sampling.advanced.reasoningBudget,
            logitBias: sampling.advanced.logitBias
        )
    }

    private static func advancedParameters(
        from sampling: SamplingParameters,
        contextOptions: ContextOptions,
        promptStyle: MLXPromptStyle
    ) -> AdvancedSamplingParameters {
        AdvancedSamplingParameters(
            minP: sampling.advanced.minP,
            typicalP: sampling.advanced.typicalP,
            topNSigma: sampling.advanced.topNSigma,
            grammar: sampling.advanced.grammar,
            mirostat: sampling.advanced.mirostat,
            xtc: sampling.advanced.xtc,
            dry: sampling.advanced.dry,
            adaptiveP: sampling.advanced.adaptiveP,
            reasoningBudget: reasoningBudget(
                for: contextOptions.reasoningLevel,
                promptStyle: promptStyle
            ),
            logitBias: sampling.advanced.logitBias
        )
    }

    private static func grammarConfiguration(
        schema: GenerationSchema?,
        toolDefinitions: [Transcript.ToolDefinition],
        promptStyle: MLXPromptStyle,
        options: GenerationOptions
    ) -> GrammarSamplingConfiguration? {
        if let schema {
            let choices = FoundationModelsSchemaSupport.stringChoices(from: schema)
            if !choices.isEmpty {
                return .choices(choices)
            }
            return .jsonSchema(FoundationModelsSchemaSupport.jsonSchemaString(from: schema))
        }
        guard requiresToolCall(options), !toolDefinitions.isEmpty else {
            return nil
        }
        return FMRequiredToolGrammarBuilder.grammar(
            from: toolDefinitions,
            promptStyle: promptStyle
        )
    }

    private static func baseSampling(
        from options: GenerationOptions,
        contextOptions: ContextOptions,
        promptStyle: MLXPromptStyle,
        fallback: SamplingParameters
    ) -> SamplingParameters {
        let values = baseSamplingValues(from: options, fallback: fallback)
        return SamplingParameters(
            temperature: values.temperature,
            topP: values.topP,
            topK: values.topK,
            repetitionPenalty: fallback.repetitionPenalty,
            frequencyPenalty: fallback.frequencyPenalty,
            presencePenalty: fallback.presencePenalty,
            repetitionPenaltyRange: fallback.repetitionPenaltyRange,
            seed: values.seed,
            stopSequences: fallback.stopSequences,
            advanced: advancedParameters(
                from: fallback,
                contextOptions: contextOptions,
                promptStyle: promptStyle
            )
        )
    }

    private static func reasoningBudget(
        for level: ContextOptions.ReasoningLevel?,
        promptStyle: MLXPromptStyle
    ) -> ReasoningBudgetConfiguration? {
        guard let level else {
            return nil
        }
        return ReasoningBudgetConfiguration(
            maximumTokens: reasoningBudgetTokenCount(for: level),
            endMarker: reasoningEndMarker(for: promptStyle)
        )
    }

    private static func reasoningBudgetTokenCount(
        for level: ContextOptions.ReasoningLevel
    ) -> Int {
        switch level {
        case .light:
            return 128

        case .moderate:
            return 512

        case .deep:
            return 2_048

        case .custom(let value):
            return Int(String(describing: value)) ?? 512

        @unknown default:
            return 512
        }
    }

    private static func reasoningEndMarker(for promptStyle: MLXPromptStyle) -> String {
        switch promptStyle {
        case .gemma:
            return "<channel|>"

        case .harmony:
            return "<|end|>"

        case .minimaxM3:
            return "</mm:think>"

        case .longCat:
            return "</longcat_think>"

        default:
            return "</think>"
        }
    }

    private static func baseSamplingValues(
        from options: GenerationOptions,
        fallback: SamplingParameters
    ) -> BaseSamplingValues {
        var values = BaseSamplingValues(
            temperature: options.temperature.map(Float.init) ?? fallback.temperature,
            topP: fallback.topP,
            topK: fallback.topK,
            seed: fallback.seed
        )
        applySamplingMode(options.samplingMode, to: &values)
        return values
    }

    private static func applySamplingMode(
        _ mode: GenerationOptions.SamplingMode?,
        to values: inout BaseSamplingValues
    ) {
        switch mode?.kind {
        case .greedy:
            values.temperature = 0
            values.topK = 1

        case let .top(value, optionSeed):
            values.topK = value
            values.seed = optionSeed.map { Int(truncatingIfNeeded: $0) }

        case let .nucleus(threshold, optionSeed):
            values.topP = Float(threshold)
            values.seed = optionSeed.map { Int(truncatingIfNeeded: $0) }

        case nil:
            return

        @unknown default:
            return
        }
    }
}
#endif
