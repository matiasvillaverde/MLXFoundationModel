import Foundation

/// Rendered prompt plus cache metadata.
public struct MLXRenderedRequest: Equatable, Hashable, Sendable {
    public let prompt: String
    public let rendererID: String
    public let cacheFingerprint: String

    public init(prompt: String, rendererID: String, cacheFingerprint: String) {
        self.prompt = prompt
        self.rendererID = rendererID
        self.cacheFingerprint = cacheFingerprint
    }
}
