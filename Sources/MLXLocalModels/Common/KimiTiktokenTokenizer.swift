import Foundation
import Tokenizers

internal final class KimiTiktokenTokenizer: @unchecked Sendable, Tokenizer {
    internal static let vocabFilename = "tiktoken.model"

    private static let pattern = try! NSRegularExpression(
        pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    )

    private let ranks: [Data: Int]
    private let byteTokensByID: [Int: Data]
    private let specialTokens: [String: Int]
    private let specialTokensByID: [Int: String]
    private let specialTokensByFirstCharacter: [Character: [String]]
    private let modelMaxLength: Int

    internal let bosToken: String? = "[BOS]"
    internal let bosTokenId: Int? = 163_584
    internal let eosToken: String? = "[EOS]"
    internal let eosTokenId: Int? = 163_585
    internal let unknownToken: String? = "[UNK]"
    internal let unknownTokenId: Int? = 163_838
    internal var fuseUnknownTokens: Bool { false }
    internal var hasChatTemplate: Bool { true }

    internal static func load(from directory: URL) throws -> KimiTiktokenTokenizer? {
        let vocabURL = directory.appending(component: vocabFilename)
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            return nil
        }

        return try KimiTiktokenTokenizer(vocabURL: vocabURL)
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

    internal init(vocabURL: URL, modelMaxLength: Int = 131_072) throws {
        self.ranks = try Self.loadRanks(from: vocabURL)
        self.byteTokensByID = Dictionary(uniqueKeysWithValues: ranks.map { ($0.value, $0.key) })
        self.specialTokens = Self.makeSpecialTokens()
        self.specialTokensByID = Dictionary(uniqueKeysWithValues: specialTokens.map { ($0.value, $0.key) })
        let orderedSpecialTokens = specialTokens.keys.sorted { lhs, rhs in
            if lhs.count != rhs.count { return lhs.count > rhs.count }
            return lhs < rhs
        }
        self.specialTokensByFirstCharacter = Dictionary(grouping: orderedSpecialTokens) { $0.first! }
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
        var rendered = try renderChat(
            messages: messages,
            chatTemplate: chatTemplate,
            addGenerationPrompt: addGenerationPrompt,
            tools: tools
        )
        var tokens = encode(text: rendered, addSpecialTokens: false)
        let limit = maxLength ?? modelMaxLength

        if truncation, tokens.count > limit {
            tokens = Array(tokens.suffix(limit))
        } else if tokens.count > limit {
            rendered = String(rendered.prefix(max(0, rendered.count - (tokens.count - limit))))
            tokens = encode(text: rendered, addSpecialTokens: false)
        }
        return tokens
    }

    private func renderChat(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        tools: [ToolSpec]?
    ) throws -> String {
        if case .literal(let template) = chatTemplate {
            return template
        }

        var result = ""
        if let tools, !tools.isEmpty {
            result += "<|im_system|>tool_declare<|im_middle|>"
            result += String(describing: tools)
            result += "<|im_end|>"
        }

        for message in messages {
            guard let role = message["role"] as? String else {
                throw Tokenizers.TokenizerError.chatTemplate("Message is missing a string role")
            }
            let roleToken = Self.roleToken(for: role)
            result += "\(roleToken)\(role)<|im_middle|>"
            result += Self.stringContent(from: message["content"])
            result += "<|im_end|>"
        }

        if addGenerationPrompt {
            result += "<|im_assistant|>assistant<|im_middle|>"
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

    private static func makeSpecialTokens() -> [String: Int] {
        let realTokens = [
            "[BOS]": 163_584,
            "[EOS]": 163_585,
            "<|im_end|>": 163_586,
            "<|im_user|>": 163_587,
            "<|im_assistant|>": 163_588,
            "<|start_header_id|>": 163_590,
            "<|end_header_id|>": 163_591,
            "[EOT]": 163_593,
            "<|im_system|>": 163_594,
            "<|tool_calls_section_begin|>": 163_595,
            "<|tool_calls_section_end|>": 163_596,
            "<|tool_call_begin|>": 163_597,
            "<|tool_call_argument_begin|>": 163_598,
            "<|tool_call_end|>": 163_599,
            "<|im_middle|>": 163_601,
            "[UNK]": 163_838,
            "[PAD]": 163_839
        ]
        let realIDs = Set(realTokens.values)
        var tokens = Dictionary(uniqueKeysWithValues: (163_584 ... 163_839).compactMap {
            realIDs.contains($0) ? nil : ("<|reserved_kimi_\($0)|>", $0)
        })
        tokens.merge(realTokens) { _, real in real }
        return tokens
    }

    private static func roleToken(for role: String) -> String {
        switch role {
        case "assistant":
            "<|im_assistant|>"
        case "system":
            "<|im_system|>"
        default:
            "<|im_user|>"
        }
    }

    private static func stringContent(from value: (any Sendable)?) -> String {
        if let string = value as? String {
            return string
        }
        if let array = value as? [[String: any Sendable]] {
            return array.compactMap { item in
                item["text"] as? String
            }.joined(separator: "\n")
        }
        return value.map { String(describing: $0) } ?? ""
    }
}
