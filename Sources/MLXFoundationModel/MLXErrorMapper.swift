#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum MLXErrorMapper {
    static func map(_ error: any Error) -> any Error {
        if let local = error as? LLMError {
            return map(local)
        }
        return error
    }

    private static func map(_ error: LLMError) -> any Error {
        switch error {
        case .rateLimitExceeded(let retryAfter):
            let resetDate = retryAfter.map { Date().addingTimeInterval(Double($0.components.seconds)) }
            return LanguageModelError.rateLimited(
                .init(resetDate: resetDate, debugDescription: String(describing: error))
            )

        case .invalidConfiguration(let message) where message.contains("image inputs"):
            return LanguageModelError.unsupportedCapability(.init(
                capability: .vision,
                debugDescription: message
            ))

        case .invalidConfiguration(let message) where message.contains("video inputs"):
            return LanguageModelError.unsupportedTranscriptContent(.init(
                unsupportedContent: [],
                debugDescription: message
            ))

        case .invalidConfiguration, .modelNotFound:
            return LanguageModelError.unsupportedTranscriptContent(
                .init(unsupportedContent: [], debugDescription: String(describing: error))
            )

        case .authenticationFailed, .networkError, .providerError:
            return error
        }
    }
}
#endif
