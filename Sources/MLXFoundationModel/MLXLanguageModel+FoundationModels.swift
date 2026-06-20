#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
extension MLXLanguageModel: LanguageModel {
    public typealias Executor = MLXExecutor

    public var capabilities: LanguageModelCapabilities {
        var values: [LanguageModelCapabilities.Capability] = []
        if model.capabilities.toolCalling {
            values.append(.toolCalling)
        }
        if supportsVisionExecution {
            values.append(.vision)
        }
        if model.capabilities.reasoning {
            values.append(.reasoning)
        }
        if model.capabilities.structuredOutput {
            values.append(.guidedGeneration)
        }
        return LanguageModelCapabilities(capabilities: values)
    }

    public var executorConfiguration: MLXExecutor.Configuration {
        MLXExecutor.Configuration(
            model: model,
            compute: compute,
            runtime: runtime,
            sampling: sampling,
            maximumResponseTokens: maximumResponseTokens
        )
    }
}
#endif
