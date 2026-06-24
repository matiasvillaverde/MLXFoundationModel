import Foundation
import Hub

/// Identifies a model plus the tokenizer and generation defaults needed before loading.
internal struct ModelConfiguration: Sendable, Equatable {
    internal enum Identifier: Sendable, Equatable {
        case id(String, revision: String)
        case directory(URL)
    }

    internal var id: Identifier

    internal var name: String {
        id.displayName
    }

    internal let tokenizerId: String?
    internal let overrideTokenizer: String?
    internal var defaultPrompt: String
    internal var extraEOSTokens: Set<String>
    internal var eosTokenIds: Set<Int>
    internal var suppressTokenIds: Set<Int>

    internal init(
        id: String,
        revision: String = "main",
        tokenizerId: String? = nil,
        overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = [],
        eosTokenIds: Set<Int> = [],
        suppressTokenIds: Set<Int> = []
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
        tokenizerId: String? = nil,
        overrideTokenizer: String? = nil,
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
            return hub.localRepoLocation(Hub.Repo(id: id))
        case .directory(let directory):
            return directory
        }
    }
}

private extension ModelConfiguration.Identifier {
    var displayName: String {
        switch self {
        case .id(let id, _):
            id
        case .directory(let url):
            "\(url.deletingLastPathComponent().lastPathComponent)/\(url.lastPathComponent)"
        }
    }
}
