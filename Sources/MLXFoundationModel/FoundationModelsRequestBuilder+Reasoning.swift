#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
extension FoundationModelsRequestBuilder {
    static func reasoningInstruction(
        for level: ContextOptions.ReasoningLevel?
    ) -> String? {
        guard let level else {
            return nil
        }
        switch level {
        case .light:
            return "Use light internal reasoning before answering."

        case .moderate:
            return "Use moderate internal reasoning before answering."

        case .deep:
            return "Use deep internal reasoning before answering."

        case .custom(let value):
            return "Use the requested internal reasoning level: \(value)."

        @unknown default:
            return nil
        }
    }

    static func reasoningOptions(
        for level: ContextOptions.ReasoningLevel?
    ) -> MLXBridgeReasoningOptions? {
        guard let level else {
            return nil
        }
        switch level {
        case .light:
            return .enabled(effort: .light)

        case .moderate:
            return .enabled(effort: .moderate)

        case .deep:
            return .enabled(effort: .deep)

        case .custom(let value):
            return .enabled(customEffort: value)

        @unknown default:
            return .enabled()
        }
    }

    static func reasoningOptions(
        for level: ContextOptions.ReasoningLevel?,
        model: MLXLanguageModel
    ) -> MLXBridgeReasoningOptions? {
        if let explicit = reasoningOptions(for: level) {
            return explicit
        }
        guard model.profile?.usesReasoningByDefault == true else {
            return nil
        }
        return .enabled()
    }
}
#endif
