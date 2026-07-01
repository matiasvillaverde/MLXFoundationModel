import Foundation
import Hub
import Tokenizers

internal struct TokenizerError: Error, Equatable, CustomStringConvertible {
    internal let message: String

    internal var description: String {
        message
    }
}

internal func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer {
    let files = try await TokenizerConfigLoader(configuration: configuration, hub: hub).load()
    if let tokenizer = files.tokenizer {
        return tokenizer
    }
    guard let tokenizerData = files.tokenizerData else {
        throw TokenizerError(message: "missing tokenizer data")
    }
    return try PreTrainedTokenizer(
        tokenizerConfig: files.tokenizerConfig,
        tokenizerData: tokenizerData
    )
}

internal func loadTokenizerConfig(configuration: ModelConfiguration, hub: HubApi) async throws -> (
    Config, Config
) {
    let files = try await TokenizerConfigLoader(configuration: configuration, hub: hub).load()
    guard let tokenizerData = files.tokenizerData else {
        throw TokenizerError(message: "missing tokenizer data")
    }
    return (files.tokenizerConfig, tokenizerData)
}

private struct TokenizerConfigFiles {
    let tokenizerConfig: Config
    let tokenizerData: Config?
    let tokenizer: (any Tokenizer)?
}

private struct TokenizerConfigLoader {
    let configuration: ModelConfiguration
    let hub: HubApi

    func load() async throws -> TokenizerConfigFiles {
        let source = try await tokenizerSource()
        do {
            if let tokenizer = try RWKV7Tokenizer.load(
                from: configuration.modelDirectory(hub: hub)
            ) {
                return TokenizerConfigFiles(
                    tokenizerConfig: Config(["tokenizer_class": Config("Rwkv7Tokenizer")]),
                    tokenizerData: nil,
                    tokenizer: tokenizer
                )
            }
            if let tokenizer = try PlamoTokenizer.load(
                from: configuration.modelDirectory(hub: hub)
            ) {
                return TokenizerConfigFiles(
                    tokenizerConfig: Config(["tokenizer_class": Config("PlamoTokenizer")]),
                    tokenizerData: nil,
                    tokenizer: tokenizer
                )
            }

            let tokenizerConfig = try await source.requiredTokenizerConfig()
            let tokenizerData = try await source.tokenizerData
            let rewriter = TokenizerConfigurationRewriter(registry: replacementTokenizers)

            return TokenizerConfigFiles(
                tokenizerConfig: rewriter.rewrite(tokenizerConfig, tokenizerData: tokenizerData),
                tokenizerData: tokenizerData,
                tokenizer: nil
            )
        } catch {
            if let tokenizer = try QwenTiktokenTokenizer.load(
                from: configuration.modelDirectory(hub: hub)
            ) {
                return TokenizerConfigFiles(
                    tokenizerConfig: Config(["tokenizer_class": Config("QwenTiktokenTokenizer")]),
                    tokenizerData: nil,
                    tokenizer: tokenizer
                )
            }
            if let tokenizer = try KimiTiktokenTokenizer.load(
                from: configuration.modelDirectory(hub: hub)
            ) {
                return TokenizerConfigFiles(
                    tokenizerConfig: Config(["tokenizer_class": Config("KimiTiktokenTokenizer")]),
                    tokenizerData: nil,
                    tokenizer: tokenizer
                )
            }
            if let tokenizer = try HunyuanTiktokenTokenizer.load(
                from: configuration.modelDirectory(hub: hub)
            ) {
                return TokenizerConfigFiles(
                    tokenizerConfig: Config([
                        "tokenizer_class": Config("HunyuanTiktokenTokenizer")
                    ]),
                    tokenizerData: nil,
                    tokenizer: tokenizer
                )
            }
            if let tokenizer = try SentencePieceModelTokenizer.load(
                from: configuration.modelDirectory(hub: hub)
            ) {
                return TokenizerConfigFiles(
                    tokenizerConfig: Config(["tokenizer_class": Config("SentencePieceModelTokenizer")]),
                    tokenizerData: nil,
                    tokenizer: tokenizer
                )
            }
            if let tokenizer = try Phi3SmallTiktokenTokenizer.load(
                from: configuration.modelDirectory(hub: hub)
            ) {
                return TokenizerConfigFiles(
                    tokenizerConfig: Config(["tokenizer_class": Config("Phi3SmallTokenizer")]),
                    tokenizerData: nil,
                    tokenizer: tokenizer
                )
            }
            throw error
        }
    }

    private func tokenizerSource() async throws -> LanguageModelConfigurationFromHub {
        switch configuration.id {
        case .directory(let directory):
            return localTokenizerSource(directory: directory)
        case .id(let id, let revision):
            return try await remoteTokenizerSource(modelID: id, revision: revision)
        }
    }

    private func remoteTokenizerSource(
        modelID: String,
        revision: String
    ) async throws -> LanguageModelConfigurationFromHub {
        let tokenizerID = configuration.tokenizerId ?? modelID
        let source = LanguageModelConfigurationFromHub(
            modelName: tokenizerID,
            revision: revision,
            hubApi: hub
        )

        do {
            _ = try await source.requiredTokenizerConfig()
            return source
        } catch where isOfflineError(error) {
            return localTokenizerSource(directory: configuration.modelDirectory(hub: hub))
        } catch {
            throw error
        }
    }

    private func localTokenizerSource(directory: URL) -> LanguageModelConfigurationFromHub {
        LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
    }

    private func isOfflineError(_ error: any Error) -> Bool {
        let nsError = error as NSError
        return nsError.domain == NSURLErrorDomain
            && nsError.code == NSURLErrorNotConnectedToInternet
    }
}

private extension LanguageModelConfigurationFromHub {
    func requiredTokenizerConfig() async throws -> Config {
        guard let tokenizerConfig = try await tokenizerConfig else {
            throw TokenizerError(message: "missing tokenizer config")
        }
        return tokenizerConfig
    }
}

internal struct TokenizerConfigurationRewriter: Sendable {
    private let registry: TokenizerReplacementRegistry

    internal init(registry: TokenizerReplacementRegistry) {
        self.registry = registry
    }

    internal func rewrite(_ tokenizerConfig: Config) -> Config {
        rewrite(tokenizerConfig, tokenizerData: nil)
    }

    internal func rewrite(_ tokenizerConfig: Config, tokenizerData: Config?) -> Config {
        let tokenizerConfig = normalizingChatTemplate(in: tokenizerConfig)

        if let tokenizerData,
           tokenizerConfig.tokenizerClass?.string() == "PreTrainedTokenizerFast",
           tokenizerData.model.type.string() == "Unigram" {
            return replacingTokenizerClass(in: tokenizerConfig, with: "XLMRobertaTokenizer")
        }

        guard let tokenizerClass = tokenizerConfig.tokenizerClass?.string(),
              let replacement = registry.replacement(for: tokenizerClass),
              replacement != tokenizerClass
        else {
            return tokenizerConfig
        }

        return replacingTokenizerClass(in: tokenizerConfig, with: replacement)
    }

    private func replacingTokenizerClass(in tokenizerConfig: Config, with replacement: String) -> Config {
        guard var dictionary = tokenizerConfig.dictionary() else {
            return tokenizerConfig
        }
        dictionary["tokenizer_class"] = Config(replacement)
        return Config(dictionary)
    }

    private func normalizingChatTemplate(in tokenizerConfig: Config) -> Config {
        guard var dictionary = tokenizerConfig.dictionary(),
              let template = dictionary["chat_template"]?.string() else {
            return tokenizerConfig
        }

        let normalizedTemplate = ChatTemplateNormalizer.normalize(template)
        guard normalizedTemplate != template else {
            return tokenizerConfig
        }

        dictionary["chat_template"] = Config(normalizedTemplate)
        return Config(dictionary)
    }
}

internal enum ChatTemplateNormalizer {
    internal static func normalize(_ template: String) -> String {
        guard template.contains("budget_reflections_v05") else {
            return template
        }

        let lines = template.components(separatedBy: .newlines)
        var rewritten: [String] = []
        var isRewritingBudgetTable = false

        for line in lines {
            if line.contains("set budget_reflections_v05 = {") {
                rewritten.append(line.replacingOccurrences(of: "= {", with: "= ["))
                isRewritingBudgetTable = true
                continue
            }

            if isRewritingBudgetTable {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed == "} -%}" {
                    rewritten.append(line.replacingOccurrences(of: "} -%}", with: "] -%}"))
                    isRewritingBudgetTable = false
                    continue
                }
                if let pair = budgetReflectionPair(from: line) {
                    rewritten.append(pair)
                    continue
                }
            }

            rewritten.append(line)
        }

        var normalized = rewritten.joined(separator: "\n")
        normalized = normalized.replacingOccurrences(
            of: "budget_reflections_v05 | dictsort",
            with: "budget_reflections_v05"
        )
        normalized = normalized.replacingOccurrences(
            of: "budget_reflections_v05[16384]",
            with: "1024"
        )
        return normalized
    }

    private static func budgetReflectionPair(from line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let hasTrailingComma = trimmed.last == ","

        let body = hasTrailingComma ? trimmed.dropLast() : trimmed[...]
        let parts = body.split(separator: ":", maxSplits: 1).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        guard parts.count == 2,
              Int(parts[0]) != nil,
              Int(parts[1]) != nil else {
            return nil
        }

        let indentation = String(line.prefix { $0 == " " || $0 == "\t" })
        return "\(indentation)[\(parts[0]), \(parts[1])]\(hasTrailingComma ? "," : "")"
    }
}

internal final class TokenizerReplacementRegistry: @unchecked Sendable {
    private static let defaultReplacements = [
        "InternLM2Tokenizer": "PreTrainedTokenizer",
        "Qwen2Tokenizer": "PreTrainedTokenizer",
        "Qwen3Tokenizer": "PreTrainedTokenizer",
        "CohereTokenizer": "PreTrainedTokenizer",
        "Ernie4_5_Tokenizer": "PreTrainedTokenizer",
        "GPTNeoXTokenizer": "PreTrainedTokenizer",
        "Rwkv7Tokenizer": "PreTrainedTokenizer",
    ]

    private let lock = NSLock()
    private var replacements: [String: String]

    internal init() {
        self.replacements = Self.defaultReplacements
    }

    internal init(replacements: [String: String]) {
        self.replacements = replacements
    }

    internal subscript(key: String) -> String? {
        get {
            replacement(for: key)
        }
        set {
            lock.withLock {
                replacements[key] = newValue
            }
        }
    }

    internal func replacement(for tokenizerClass: String) -> String? {
        lock.withLock {
            replacements[tokenizerClass]
        }
    }
}

internal let replacementTokenizers = TokenizerReplacementRegistry()

internal protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)
}

internal struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    private let tokenizer: Tokenizer
    private var segmentTokens: [Int] = []
    private var emittedSegment = ""

    internal init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    internal mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    internal mutating func next() -> String? {
        let decodedSegment = tokenizer.decode(tokens: segmentTokens)
        let newText = newSuffix(in: decodedSegment)

        guard newText.last != "\u{fffd}" else {
            return nil
        }

        if decodedSegment.hasSuffix("\n") {
            startNewSegment()
        } else {
            emittedSegment = decodedSegment
        }

        return String(newText)
    }

    private func newSuffix(in decodedSegment: String) -> Substring {
        guard decodedSegment.hasPrefix(emittedSegment) else {
            return decodedSegment[...]
        }
        return decodedSegment.dropFirst(emittedSegment.count)
    }

    private mutating func startNewSegment() {
        guard let lastToken = segmentTokens.last else {
            emittedSegment = ""
            return
        }

        segmentTokens = [lastToken]
        emittedSegment = tokenizer.decode(tokens: segmentTokens)
    }
}
