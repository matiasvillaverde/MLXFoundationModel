// Copyright © 2024 Apple Inc.

import Foundation
import Hub

/// Configuration for a given model name with overrides for prompts and tokens.
///
/// See e.g. `MLXLM.ModelRegistry` for an example of use.
internal struct ModelConfiguration: Sendable {

    internal enum Identifier: Sendable {
        case id(String, revision: String = "main")
        case directory(URL)
    }

    internal var id: Identifier

    internal var name: String {
        switch id {
        case .id(let id, _):
            id
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// pull the tokenizer from an alternate id
    internal let tokenizerId: String?

    /// overrides for TokenizerModel/knownTokenizers -- useful before swift-transformers is updated
    internal let overrideTokenizer: String?

    /// A reasonable default prompt for the model
    internal var defaultPrompt: String

    /// Additional tokens to use for end of string
    internal var extraEOSTokens: Set<String>

    /// Additional token ids loaded from model config or generation_config.json.
    internal var eosTokenIds: Set<Int>

    /// Token ids that must be suppressed during sampling.
    internal var suppressTokenIds: Set<Int>

    internal init(
        id: String, revision: String = "main",
        tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = [],
        eosTokenIds: Set<Int> = [],
        suppressTokenIds: Set<Int> = [],
        preparePrompt: (@Sendable (String) -> String)? = nil
    ) {
        self.id = .id(id, revision: revision)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.suppressTokenIds = suppressTokenIds
    }

    internal init(
        directory: URL,
        tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = ["<end_of_turn>"],
        eosTokenIds: Set<Int> = [],
        suppressTokenIds: Set<Int> = []
    ) {
        self.id = .directory(directory)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.suppressTokenIds = suppressTokenIds
    }

    internal func modelDirectory(hub: HubApi = HubApi()) -> URL {
        switch id {
        case .id(let id, _):
            // download the model weights and config
            let repo = Hub.Repo(id: id)
            return hub.localRepoLocation(repo)

        case .directory(let directory):
            return directory
        }
    }
}

extension ModelConfiguration: Equatable {

}

extension ModelConfiguration.Identifier: Equatable {

    internal static func == (lhs: ModelConfiguration.Identifier, rhs: ModelConfiguration.Identifier)
        -> Bool
    {
        switch (lhs, rhs) {
        case (.id(let lhsID, let lhsRevision), .id(let rhsID, let rhsRevision)):
            lhsID == rhsID && lhsRevision == rhsRevision
        case (.directory(let lhsURL), .directory(let rhsURL)):
            lhsURL == rhsURL
        default:
            false
        }
    }
}
