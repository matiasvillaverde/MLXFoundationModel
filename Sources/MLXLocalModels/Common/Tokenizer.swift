// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

struct TokenizerError: Error {
    let message: String
}

internal func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
        configuration: configuration, hub: hub)

    return try PreTrainedTokenizer(
        tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

internal func loadTokenizerConfig(configuration: ModelConfiguration, hub: HubApi) async throws -> (
    Config, Config
) {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config: LanguageModelConfigurationFromHub

    switch configuration.id {
    case .id(let id, let revision):
        do {
            // the load can fail (async when we try to use it)
            let loaded = LanguageModelConfigurationFromHub(
                modelName: configuration.tokenizerId ?? id, revision: revision, hubApi: hub)
            _ = try await loaded.tokenizerConfig
            config = loaded
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain
                && nserror.code == NSURLErrorNotConnectedToInternet
            {
                // Internet connection appears to be offline -- fall back to loading from
                // the local directory
                config = LanguageModelConfigurationFromHub(
                    modelFolder: configuration.modelDirectory(hub: hub), hubApi: hub)
            } else {
                throw error
            }
        }
    case .directory(let directory):
        config = LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
    }

    guard var tokenizerConfig = try await config.tokenizerConfig else {
        throw TokenizerError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData

    tokenizerConfig = updateTokenizerConfig(tokenizerConfig)

    return (tokenizerConfig, tokenizerData)
}

private func updateTokenizerConfig(_ tokenizerConfig: Config) -> Config {
    // Workaround: replacement tokenizers for unhandled values in swift-transformers
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.string(),
        let replacement = replacementTokenizers[tokenizerClass]
    {
        if var dictionary = tokenizerConfig.dictionary() {
            dictionary["tokenizer_class"] = .init(replacement)
            return Config(dictionary)
        }
    }
    return tokenizerConfig
}

/// Thread-safe registry for tokenizer class replacements
///
/// This class is marked `@unchecked Sendable` because:
/// - It contains mutable state (`replacementTokenizers` dictionary) protected by explicit synchronization (`NSLock`)
/// - All mutations are guarded by lock-protected critical sections
/// - The lock ensures that all dictionary accesses are properly serialized
///
/// Safety guarantees:
/// - Atomic operations: All reads and writes are protected by NSLock
/// - Thread-safe access: Multiple threads can safely query and modify replacements
/// - No data races: The lock serializes all access to the mutable dictionary
/// - Small critical sections: Lock contention is minimal due to short operations
internal class TokenizerReplacementRegistry: @unchecked Sendable {

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention. this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    /// overrides for TokenizerModel/knownTokenizers
    private var replacementTokenizers = [
        "InternLM2Tokenizer": "PreTrainedTokenizer",
        "Qwen2Tokenizer": "PreTrainedTokenizer",
        "Qwen3Tokenizer": "PreTrainedTokenizer",
        "CohereTokenizer": "PreTrainedTokenizer",
    ]

    public subscript(key: String) -> String? {
        get {
            lock.withLock {
                replacementTokenizers[key]
            }
        }
        set {
            lock.withLock {
                replacementTokenizers[key] = newValue
            }
        }
    }
}

internal let replacementTokenizers = TokenizerReplacementRegistry()

internal protocol StreamingDetokenizer: IteratorProtocol<String> {

    mutating func append(token: Int)

}

internal struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    internal init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating internal func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokens: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokens: segmentTokens)
        let new = newSegment.suffix(newSegment.count - segment.count)

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if new.last == "\u{fffd}" {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return String(new)
    }

}
