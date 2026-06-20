import Foundation

extension MLXSession {
    nonisolated static func promptCacheVariant(
        for speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    ) -> String? {
        speculativeDecoding.map { "speculative:\($0.draftContext.configuration.name)" }
    }

    nonisolated func isTimedOut(_ genContext: GenerationContext) -> Bool {
        guard let maxTime: Duration = genContext.input.limits.maxTime else {
            return false
        }
        return genContext.generationStartTime.duration(to: genContext.clock.now) >= maxTime
    }
}
