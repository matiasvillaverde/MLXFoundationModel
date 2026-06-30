import Foundation
import Tokenizers

internal final class Phi3SmallTiktokenTokenizer: @unchecked Sendable, Tokenizer {
    internal static let vocabFilename = "cl100k_base.tiktoken"

    private static let pattern = try! NSRegularExpression(
        pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    )

    private static let defaultChatTemplate =
        "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\\n' + message['content'] + '<|end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% else %}{{ eos_token }}{% endif %}"

    private let ranks: [Data: Int]
    private let byteTokensByID: [Int: Data]
    private let specialTokens: [String: Int]
    private let specialTokensByID: [Int: String]
    private let orderedSpecialTokens: [String]
    private let chatTemplate: String
    private let modelMaxLength: Int

    internal let bosToken: String? = "<|endoftext|>"
    internal let bosTokenId: Int? = 100_257
    internal let eosToken: String? = "<|endoftext|>"
    internal let eosTokenId: Int? = 100_257
    internal let unknownToken: String? = nil
    internal let unknownTokenId: Int? = nil
    internal var fuseUnknownTokens: Bool { false }
    internal var hasChatTemplate: Bool { true }

    internal static func load(from directory: URL) throws -> Phi3SmallTiktokenTokenizer? {
        let vocabURL = directory.appending(component: vocabFilename)
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            return nil
        }

        let tokenizerConfigURL = directory.appending(component: "tokenizer_config.json")
        let config = try? Phi3SmallTokenizerConfig.load(from: tokenizerConfigURL)
        return try Phi3SmallTiktokenTokenizer(
            vocabURL: vocabURL,
            chatTemplate: config?.chatTemplate ?? defaultChatTemplate,
            modelMaxLength: config?.modelMaxLength ?? 8_192
        )
    }

    internal static func vocabularyEntries(from vocabURL: URL) throws -> [Int: String] {
        let ranks = try loadRanks(from: vocabURL)
        var entries = ranks.reduce(into: [Int: String]()) { result, element in
            result[element.value] = String(decoding: element.key, as: UTF8.self)
        }
        for (token, id) in makeSpecialTokens() {
            entries[id] = token
        }
        return entries
    }

    internal init(
        vocabURL: URL,
        chatTemplate: String = Phi3SmallTiktokenTokenizer.defaultChatTemplate,
        modelMaxLength: Int = 8_192
    ) throws {
        self.ranks = try Self.loadRanks(from: vocabURL)
        self.byteTokensByID = Dictionary(uniqueKeysWithValues: self.ranks.map { ($0.value, $0.key) })
        self.specialTokens = Self.makeSpecialTokens()
        self.specialTokensByID = Dictionary(uniqueKeysWithValues: specialTokens.map { ($0.value, $0.key) })
        self.orderedSpecialTokens = specialTokens.keys.sorted { lhs, rhs in
            if lhs.count != rhs.count { return lhs.count > rhs.count }
            return lhs < rhs
        }
        self.chatTemplate = chatTemplate
        self.modelMaxLength = modelMaxLength
    }

    internal func tokenize(text: String) -> [String] {
        encode(text: text, addSpecialTokens: false).compactMap(convertIdToToken)
    }

    internal func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    internal func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens: [Int] = []
        for segment in splitSpecialSegments(text) {
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
                if !skipSpecialTokens {
                    flushBytes()
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
        if let special = specialTokens[token] {
            return special
        }
        return ranks[Data(token.utf8)]
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
                "Phi3SmallTokenizer does not define a tool-use template"
            )
        }
        if additionalContext?.isEmpty == false {
            throw Tokenizers.TokenizerError.chatTemplate(
                "Phi3SmallTokenizer does not use additional template context"
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
        case .literal(let template):
            guard template == chatTemplate || template == Self.defaultChatTemplate else {
                throw Tokenizers.TokenizerError.chatTemplate(
                    "Phi3SmallTokenizer only supports its configured chat template"
                )
            }
        case .name(let name):
            guard name == "default" else {
                throw Tokenizers.TokenizerError.chatTemplate(
                    "No chat template named \"\(name)\" was found"
                )
            }
        }
    }

    private func render(messages: [Message], addGenerationPrompt: Bool) throws -> String {
        var result = bosToken ?? ""
        for message in messages {
            guard let role = message["role"] as? String else {
                throw Tokenizers.TokenizerError.chatTemplate("Message is missing a string role")
            }
            let content = Self.stringContent(from: message["content"])
            result += "<|\(role)|>\n\(content)<|end|>\n"
        }

        if addGenerationPrompt {
            result += "<|assistant|>\n"
        } else if let eosToken {
            result += eosToken
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
            if let special = orderedSpecialTokens.first(where: { text[cursor...].hasPrefix($0) }) {
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
        let contents = try String(contentsOf: vocabURL, encoding: .utf8)
        for line in contents.split(whereSeparator: \.isNewline) where !line.isEmpty {
            let parts = line.split(separator: " ")
            guard parts.count == 2,
                  let token = Data(base64Encoded: String(parts[0])),
                  let rank = Int(parts[1])
            else {
                throw Tokenizers.TokenizerError.malformedVocab
            }
            ranks[token] = rank
        }
        return ranks
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

    private static func makeSpecialTokens() -> [String: Int] {
        var tokens: [String: Int] = [
            "<|dummy_id_2|>": 100_256,
            "<|endoftext|>": 100_257,
            "<|fim_prefix|>": 100_258,
            "<|fim_middle|>": 100_259,
            "<|fim_suffix|>": 100_260,
            "<|system|>": 100_261,
            "<|user|>": 100_262,
            "<|assistant|>": 100_263,
            "<|dummy_id_0|>": 100_264,
            "<|dummy_id_1|>": 100_265,
            "<|end|>": 100_266,
            "<|endofprompt|>": 100_276
        ]

        for offset in 3 ... 11 {
            tokens["<|dummy_id_\(offset)|>"] = 100_264 + offset
        }
        for offset in 12 ... 86 {
            tokens["<|dummy_id_\(offset)|>"] = 100_265 + offset
        }
        return tokens
    }
}

private struct Phi3SmallTokenizerConfig: Decodable {
    let chatTemplate: String?
    let modelMaxLength: Int?

    enum CodingKeys: String, CodingKey {
        case chatTemplate = "chat_template"
        case modelMaxLength = "model_max_length"
    }

    static func load(from url: URL) throws -> Self? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try JSONDecoder.json5().decode(Self.self, from: Data(contentsOf: url))
    }
}
