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
        from options: GenerationOptions,
        schema: GenerationSchema?,
        toolDefinitions: [Transcript.ToolDefinition],
        fallback: SamplingParameters
    ) -> SamplingParameters {
        let sampling = baseSampling(from: options, fallback: fallback)
        guard let grammar = grammarConfiguration(
            schema: schema,
            toolDefinitions: toolDefinitions,
            options: options
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
            logitBias: sampling.advanced.logitBias
        )
    }

    private static func grammarConfiguration(
        schema: GenerationSchema?,
        toolDefinitions: [Transcript.ToolDefinition],
        options: GenerationOptions
    ) -> GrammarSamplingConfiguration? {
        if let schema {
            return .jsonSchema(FoundationModelsSchemaSupport.jsonSchemaString(from: schema))
        }
        guard requiresToolCall(options), !toolDefinitions.isEmpty else {
            return nil
        }
        return .jsonSchema(
            FoundationModelsToolSchemaBuilder.requiredToolCallSchema(from: toolDefinitions)
        )
    }

    private static func baseSampling(
        from options: GenerationOptions,
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
            advanced: fallback.advanced
        )
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
