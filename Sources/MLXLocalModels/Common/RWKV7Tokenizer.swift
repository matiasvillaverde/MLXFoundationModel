import Foundation
import Tokenizers

internal final class RWKV7Tokenizer: @unchecked Sendable, Tokenizer {
    internal static let vocabFilename = "vocab.json"

    private final class TrieNode {
        var tokenID: Int?
        var children: [UInt8: TrieNode] = [:]
    }

    private let root = TrieNode()
    private let tokenBytesByID: [Int: [UInt8]]
    private let tokenStringsByID: [Int: String]
    private let tokenIDsByString: [String: Int]
    private let specialTokens: [String: Int]
    private let specialTokensByID: [Int: String]
    private let specialTokensByFirstCharacter: [Character: [String]]

    internal let bosToken: String? = "<|endoftext|>"
    internal let bosTokenId: Int? = 0
    internal let eosToken: String? = "<|endoftext|>"
    internal let eosTokenId: Int? = 0
    internal let unknownToken: String? = "<|endoftext|>"
    internal let unknownTokenId: Int? = 0
    internal var fuseUnknownTokens: Bool { false }
    internal var hasChatTemplate: Bool { false }

    internal static func load(from directory: URL) throws -> RWKV7Tokenizer? {
        let vocabURL = directory.appending(component: vocabFilename)
        guard FileManager.default.fileExists(atPath: vocabURL.path),
              canLoad(from: directory)
        else {
            return nil
        }
        return try RWKV7Tokenizer(vocabURL: vocabURL)
    }

    internal static func canLoad(from directory: URL) -> Bool {
        isRWKV7Tokenizer(directory: directory)
    }

    internal static func vocabularyEntries(from vocabURL: URL) throws -> [Int: String] {
        let entries = try loadVocabulary(from: vocabURL)
        return Dictionary(uniqueKeysWithValues: entries.map { token, id in
            (id, decodeTokenBytes(Self.byteString(from: token)))
        })
    }

    internal init(vocabURL: URL) throws {
        let vocabulary = try Self.loadVocabulary(from: vocabURL)
        self.tokenIDsByString = vocabulary
        self.tokenBytesByID = Dictionary(uniqueKeysWithValues: vocabulary.map { token, id in
            (id, Self.byteString(from: token))
        })
        self.tokenStringsByID = Dictionary(uniqueKeysWithValues: vocabulary.map { token, id in
            (id, token)
        })
        self.specialTokens = ["<|endoftext|>": 0]
        self.specialTokensByID = [0: "<|endoftext|>"]
        self.specialTokensByFirstCharacter = Dictionary(grouping: Array(specialTokens.keys)) {
            $0.first!
        }

        for (token, id) in vocabulary where token != "<|endoftext|>" {
            insert(Self.byteString(from: token), tokenID: id)
        }
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
                appendByteLevelTokens(for: value, to: &tokens)
            }
        }
        return tokens
    }

    internal func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(tokens.count)

        for token in tokens {
            if let special = specialTokensByID[token] {
                if !skipSpecialTokens {
                    bytes.append(contentsOf: special.utf8)
                }
                continue
            }
            guard let tokenBytes = tokenBytesByID[token] else {
                continue
            }
            bytes.append(contentsOf: tokenBytes)
        }

        return String(decoding: bytes, as: UTF8.self)
    }

    internal func convertTokenToId(_ token: String) -> Int? {
        tokenIDsByString[token] ?? specialTokens[token]
    }

    internal func convertIdToToken(_ id: Int) -> String? {
        specialTokensByID[id] ?? tokenStringsByID[id]
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

    internal func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
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
        throw Tokenizers.TokenizerError.missingChatTemplate
    }

    private func insert(_ bytes: [UInt8], tokenID: Int) {
        var node = root
        for byte in bytes {
            if node.children[byte] == nil {
                node.children[byte] = TrieNode()
            }
            node = node.children[byte]!
        }
        node.tokenID = tokenID
    }

    private func appendByteLevelTokens(for text: String, to tokens: inout [Int]) {
        let bytes = Array(text.utf8)
        var offset = 0
        while offset < bytes.count {
            var node = root
            var cursor = offset
            var best: (end: Int, tokenID: Int)?
            while cursor < bytes.count, let next = node.children[bytes[cursor]] {
                node = next
                cursor += 1
                if let tokenID = node.tokenID {
                    best = (cursor, tokenID)
                }
            }

            if let best {
                tokens.append(best.tokenID)
                offset = best.end
            } else {
                let scalar = UnicodeScalar(Int(bytes[offset]))!
                let token = String(scalar)
                tokens.append(tokenIDsByString[token] ?? 0)
                offset += 1
            }
        }
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

    private static func loadVocabulary(from vocabURL: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: vocabURL)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Tokenizers.TokenizerError.malformedVocab
        }

        var entries: [String: Int] = [:]
        entries.reserveCapacity(object.count)
        for (token, value) in object {
            if let id = value as? Int {
                entries[token] = id
            } else if let number = value as? NSNumber {
                entries[token] = number.intValue
            } else {
                throw Tokenizers.TokenizerError.malformedVocab
            }
        }
        return entries
    }

    private static func isRWKV7Tokenizer(directory: URL) -> Bool {
        let tokenizerJSON = directory.appending(component: "tokenizer.json")
        if let data = try? Data(contentsOf: tokenizerJSON),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let model = json["model"] as? [String: Any],
           model["type"] as? String == "RWKV7LongestMatch" {
            return true
        }

        let tokenizerConfig = directory.appending(component: "tokenizer_config.json")
        if let data = try? Data(contentsOf: tokenizerConfig),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           json["tokenizer_class"] as? String == "Rwkv7Tokenizer" {
            return true
        }

        return false
    }

    private static func byteString(from token: String) -> [UInt8] {
        token.unicodeScalars.map { UInt8($0.value & 0xff) }
    }

    private static func decodeTokenBytes(_ bytes: [UInt8]) -> String {
        String(decoding: bytes, as: UTF8.self)
    }
}
