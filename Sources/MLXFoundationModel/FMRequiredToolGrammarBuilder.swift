#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FMRequiredToolGrammarBuilder {
    static func grammar(
        from definitions: [Transcript.ToolDefinition],
        promptStyle: MLXPromptStyle
    ) -> GrammarSamplingConfiguration {
        MLXRequiredToolGrammarBuilder.grammar(
            from: definitions.map(FoundationModelsToolSchemaBuilder.bridgeToolDefinition),
            promptStyle: promptStyle
        )
    }
}
#endif
