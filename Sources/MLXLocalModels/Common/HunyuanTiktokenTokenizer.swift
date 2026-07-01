import Foundation
import Tokenizers

internal final class HunyuanTiktokenTokenizer: @unchecked Sendable, Tokenizer {
    internal static let vocabFilename = "hy.tiktoken"

    private static let pattern = try! NSRegularExpression(
        pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    )

    private let ranks: [Data: Int]
    private let byteTokensByID: [Int: Data]
    private let specialTokens: [String: Int]
    private let specialTokensByID: [Int: String]
    private let specialTokensByFirstCharacter: [Character: [String]]
    private let modelMaxLength: Int

    internal let bosToken: String? = "<|bos|>"
    internal let bosTokenId: Int?
    internal let eosToken: String? = "<|eos|>"
    internal let eosTokenId: Int?
    internal let unknownToken: String? = nil
    internal let unknownTokenId: Int? = nil
    internal var fuseUnknownTokens: Bool { false }
    internal var hasChatTemplate: Bool { true }

    internal static func load(from directory: URL) throws -> HunyuanTiktokenTokenizer? {
        let vocabURL = directory.appending(component: vocabFilename)
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            return nil
        }

        let config = try HunyuanTokenizerConfig.load(
            from: directory.appending(component: "tokenizer_config.json")
        )
        return try HunyuanTiktokenTokenizer(
            vocabURL: vocabURL,
            modelMaxLength: config.modelMaxLength ?? 1_048_576
        )
    }

    internal static func vocabularyEntries(from vocabURL: URL) throws -> [Int: String] {
        let ranks = try loadRanks(from: vocabURL)
        var entries = ranks.reduce(into: [Int: String]()) { result, element in
            result[element.value] = String(decoding: element.key, as: UTF8.self)
        }
        for (token, id) in makeSpecialTokens(mergeableCount: ranks.count) {
            entries[id] = token
        }
        return entries
    }

    internal init(vocabURL: URL, modelMaxLength: Int = 1_048_576) throws {
        self.ranks = try Self.loadRanks(from: vocabURL)
        self.byteTokensByID = Dictionary(uniqueKeysWithValues: ranks.map { ($0.value, $0.key) })
        self.specialTokens = Self.makeSpecialTokens(mergeableCount: ranks.count)
        self.specialTokensByID = Dictionary(
            uniqueKeysWithValues: specialTokens.map { ($0.value, $0.key) }
        )
        let orderedSpecialTokens = specialTokens.keys.sorted { lhs, rhs in
            if lhs.count != rhs.count {
                return lhs.count > rhs.count
            }
            return lhs < rhs
        }
        self.specialTokensByFirstCharacter = Dictionary(grouping: orderedSpecialTokens) {
            $0.first!
        }
        self.modelMaxLength = modelMaxLength
        self.bosTokenId = specialTokens["<|bos|>"]
        self.eosTokenId = specialTokens["<|eos|>"]
    }

    internal func tokenize(text: String) -> [String] {
        encode(text: text, addSpecialTokens: false).compactMap(convertIdToToken)
    }

    internal func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    internal func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let normalizedText = text.precomposedStringWithCanonicalMapping
        var tokens: [Int] = []
        for segment in splitSpecialSegments(normalizedText) {
            switch segment {
            case .special(let token):
                if let id = specialTokens[token] {
                    tokens.append(id)
                }
            case .text(let value):
                appendOrdinaryText(value, to: &tokens)
            }
        }
        return tokens
    }

    internal func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        var text = ""
        var bytes = Data()

        func flushBytes() {
            guard !bytes.isEmpty else { return }
            text += String(decoding: bytes, as: UTF8.self)
            bytes.removeAll(keepingCapacity: true)
        }

        for token in tokens {
            if let special = specialTokensByID[token] {
                flushBytes()
                if !skipSpecialTokens {
                    text += special
                }
                continue
            }

            guard let tokenBytes = byteTokensByID[token] else {
                continue
            }
            bytes.append(tokenBytes)
        }

        flushBytes()
        return text
    }

    internal func convertTokenToId(_ token: String) -> Int? {
        specialTokens[token] ?? ranks[Data(token.utf8)]
    }

    internal func convertIdToToken(_ id: Int) -> String? {
        if let special = specialTokensByID[id] {
            return special
        }
        guard let bytes = byteTokensByID[id] else {
            return nil
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    internal func applyChatTemplate(messages: [Message]) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil
        )
    }

    internal func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: tools
        )
    }

    internal func applyChatTemplate(
        messages: [Message],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: tools,
            additionalContext: additionalContext
        )
    }

    internal func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: chatTemplate,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil
        )
    }

    internal func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: .literal(chatTemplate),
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil
        )
    }

    internal func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: chatTemplate,
            addGenerationPrompt: addGenerationPrompt,
            truncation: truncation,
            maxLength: maxLength,
            tools: tools,
            additionalContext: nil
        )
    }

    internal func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        if let tools, !tools.isEmpty {
            throw Tokenizers.TokenizerError.chatTemplate(
                "Hunyuan tokenizer does not define a tool-use template"
            )
        }
        if additionalContext?.isEmpty == false {
            throw Tokenizers.TokenizerError.chatTemplate(
                "Hunyuan tokenizer does not use additional template context"
            )
        }
        try validate(chatTemplate)

        let rendered = try render(messages: messages, addGenerationPrompt: addGenerationPrompt)
        var encoded = encode(text: rendered, addSpecialTokens: false)
        let limit = min(maxLength ?? encoded.count, modelMaxLength)
        if encoded.count > limit, truncation {
            encoded = Array(encoded.prefix(limit))
        }
        return encoded
    }

    private func validate(_ argument: ChatTemplateArgument?) throws {
        guard let argument else { return }
        switch argument {
        case .literal:
            return
        case .name(let name):
            guard name == "default" else {
                throw Tokenizers.TokenizerError.chatTemplate(
                    "No chat template named \"\(name)\" was found"
                )
            }
        }
    }

    private func render(messages: [Message], addGenerationPrompt: Bool) throws -> String {
        var result = ""
        var hasHead = true

        for (index, message) in messages.enumerated() {
            guard let role = message["role"] as? String else {
                throw Tokenizers.TokenizerError.chatTemplate("Message is missing a string role")
            }
            var content = Self.stringContent(from: message["content"])

            if index == 0 {
                if content.isEmpty {
                    hasHead = false
                } else {
                    content = "<|startoftext|>\(content)<|extra_4|>"
                }
            }

            if role == "user" {
                if index == 1 && !hasHead {
                    content = "<|startoftext|>\(content)"
                }
                if index == 1 && hasHead {
                    content += "<|extra_0|>"
                } else {
                    content = "<|startoftext|>\(content)<|extra_0|>"
                }
            } else if role == "assistant" {
                content += "<|eos|>"
            }

            result += content
        }

        return result
    }

    private func appendOrdinaryText(_ text: String, to tokens: inout [Int]) {
        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)
        Self.pattern.enumerateMatches(in: text, range: range) { match, _, _ in
            guard let match else { return }
            let piece = nsText.substring(with: match.range)
            tokens.append(contentsOf: bytePairEncode(Data(piece.utf8)))
        }
    }

    private func bytePairEncode(_ bytes: Data) -> [Int] {
        if let rank = ranks[bytes] {
            return [rank]
        }

        var parts = bytes.map { Data([$0]) }
        guard parts.count > 1 else {
            return parts.compactMap { ranks[$0] }
        }

        while let merge = nextMerge(in: parts) {
            parts[merge.index].append(parts[merge.index + 1])
            parts.remove(at: merge.index + 1)
        }

        return parts.compactMap { ranks[$0] }
    }

    private func nextMerge(in parts: [Data]) -> (index: Int, rank: Int)? {
        guard parts.count > 1 else { return nil }

        var best: (index: Int, rank: Int)?
        for index in 0 ..< (parts.count - 1) {
            var pair = Data()
            pair.reserveCapacity(parts[index].count + parts[index + 1].count)
            pair.append(parts[index])
            pair.append(parts[index + 1])

            guard let rank = ranks[pair] else { continue }
            if best.map({ rank < $0.rank }) ?? true {
                best = (index, rank)
            }
        }
        return best
    }

    private enum Segment {
        case text(String)
        case special(String)
    }

    private func splitSpecialSegments(_ text: String) -> [Segment] {
        var segments: [Segment] = []
        var cursor = text.startIndex
        var plainStart = cursor

        while cursor < text.endIndex {
            let candidates = specialTokensByFirstCharacter[text[cursor]] ?? []
            if let special = candidates.first(where: { text[cursor...].hasPrefix($0) }) {
                if plainStart < cursor {
                    segments.append(.text(String(text[plainStart ..< cursor])))
                }
                segments.append(.special(special))
                cursor = text.index(cursor, offsetBy: special.count)
                plainStart = cursor
            } else {
                cursor = text.index(after: cursor)
            }
        }

        if plainStart < text.endIndex {
            segments.append(.text(String(text[plainStart ..< text.endIndex])))
        }
        return segments
    }

    private static func loadRanks(from vocabURL: URL) throws -> [Data: Int] {
        var ranks: [Data: Int] = [:]
        var nextRank = 0
        let contents = try String(contentsOf: vocabURL, encoding: .utf8)
        for line in contents.split(whereSeparator: \.isNewline) where !line.isEmpty {
            let parts = line.split(separator: " ")
            guard parts.count == 2,
                  let token = Data(base64Encoded: String(parts[0]))
            else {
                throw Tokenizers.TokenizerError.malformedVocab
            }
            guard ranks[token] == nil else {
                continue
            }
            ranks[token] = nextRank
            nextRank += 1
        }
        return ranks
    }

    private static func makeSpecialTokens(mergeableCount: Int) -> [String: Int] {
        var tokens = [
            "<|endoftext|>": mergeableCount,
            "<|startoftext|>": mergeableCount + 1,
            "<|bos|>": mergeableCount + 2,
            "<|eos|>": mergeableCount + 3,
            "<|pad|>": mergeableCount + 4
        ]
        for index in 0 ..< 205 {
            tokens["<|extra_\(index)|>"] = mergeableCount + 5 + index
        }
        return tokens
    }

    private static func stringContent(from value: (any Sendable)?) -> String {
        switch value {
        case let string as String:
            return string
        case let substring as Substring:
            return String(substring)
        case let convertible as CustomStringConvertible:
            return convertible.description
        case .some(let value):
            return "\(value)"
        case nil:
            return ""
        }
    }
}

private struct HunyuanTokenizerConfig: Decodable {
    let modelMaxLength: Int?

    enum CodingKeys: String, CodingKey {
        case modelMaxLength = "model_max_length"
    }

    static func load(from url: URL) throws -> Self {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return Self(modelMaxLength: nil)
        }
        return try JSONDecoder.json5().decode(Self.self, from: Data(contentsOf: url))
    }
}
