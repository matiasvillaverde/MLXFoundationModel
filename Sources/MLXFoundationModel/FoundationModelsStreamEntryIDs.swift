#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
struct FoundationModelsStreamEntryIDs: Sendable {
    let response: String
    let reasoning: String
    let toolCalls: String
}
#endif
