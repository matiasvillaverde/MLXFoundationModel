import Foundation
import Tokenizers

internal final class PlamoTokenizer: @unchecked Sendable, Tokenizer {
    internal static let vocabFilename = "tokenizer.jsonl"

    private struct VocabularyEntry {
        let token: String
        let score: Int
        let kind: String
    }

    private struct TrieMatch {
        let tokenID: Int
        let score: Int
        let length: Int
    }

    private final class TrieNode {
        var match: TrieMatch?
        var children: [UnicodeScalar: TrieNode] = [:]
    }

    private struct EncodingStep {
        let nextIndex: Int
        let tokenIDs: [Int]
        let length: Int
    }

    private let root = TrieNode()
    private let entries: [VocabularyEntry]
    private let tokenIDsByString: [String: Int]
    private let byteTokenIDsByByte: [UInt8: Int]
    private let byteValuesByTokenID: [Int: UInt8]
    private let specialTokens: [String: Int]
    private let specialTokensByID: [Int: String]
    private let specialTokensByFirstCharacter: [Character: [String]]
    private let addBosToken: Bool
    private let addEosToken: Bool

    internal var bosToken: String? { "<|plamo:bos|>" }
    internal var bosTokenId: Int? { specialTokens["<|plamo:bos|>"] }
    internal var eosToken: String? { "<|plamo:eos|>" }
    internal var eosTokenId: Int? { specialTokens["<|plamo:eos|>"] }
    internal var unknownToken: String? { "<|plamo:unk|>" }
    internal var unknownTokenId: Int? { specialTokens["<|plamo:unk|>"] }
    internal var fuseUnknownTokens: Bool { false }
    internal var hasChatTemplate: Bool { false }

    internal static func load(from directory: URL) throws -> PlamoTokenizer? {
        let vocabURL = directory.appending(component: vocabFilename)
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            return nil
        }

        let config = PlamoTokenizerConfig.load(
            from: directory.appending(component: "tokenizer_config.json")
        )
        return try PlamoTokenizer(
            vocabURL: vocabURL,
            addBosToken: config.addBosToken,
            addEosToken: config.addEosToken
        )
    }

    internal static func vocabularyEntries(from vocabURL: URL) throws -> [Int: String] {
        let entries = try loadEntries(from: vocabURL)
        return Dictionary(uniqueKeysWithValues: entries.enumerated().map { id, entry in
            (id, decodedVocabularyToken(entry))
        })
    }

    internal init(
        vocabURL: URL,
        addBosToken: Bool = true,
        addEosToken: Bool = false
    ) throws {
        self.entries = try Self.loadEntries(from: vocabURL)
        self.tokenIDsByString = Dictionary(uniqueKeysWithValues: entries.enumerated().map {
            ($0.element.token, $0.offset)
        })
        self.byteTokenIDsByByte = Self.byteTokenIDsByByte(entries)
        self.byteValuesByTokenID = Dictionary(uniqueKeysWithValues: byteTokenIDsByByte.map {
            ($0.value, $0.key)
        })
        self.specialTokens = Self.makeSpecialTokens(from: tokenIDsByString)
        self.specialTokensByID = Dictionary(uniqueKeysWithValues: specialTokens.map {
            ($0.value, $0.key)
        })
        let orderedSpecialTokens = specialTokens.keys.sorted { lhs, rhs in
            if lhs.count != rhs.count { return lhs.count > rhs.count }
            return lhs < rhs
        }
        self.specialTokensByFirstCharacter = Dictionary(grouping: orderedSpecialTokens) {
            $0.first!
        }
        self.addBosToken = addBosToken
        self.addEosToken = addEosToken

        for (tokenID, entry) in entries.enumerated() where entry.kind == "NORMAL" {
            insert(entry.token, tokenID: tokenID, score: entry.score)
        }
    }

    internal func tokenize(text: String) -> [String] {
        encode(text: text, addSpecialTokens: false).compactMap(convertIdToToken)
    }

    internal func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    internal func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokenIDs: [Int] = []
        if addSpecialTokens, addBosToken, let bosTokenId {
            tokenIDs.append(bosTokenId)
        }

        for segment in splitSpecialSegments(text) {
            switch segment {
            case .special(let token):
                if let id = specialTokens[token] {
                    tokenIDs.append(id)
                }
            case .text(let value):
                tokenIDs.append(contentsOf: encodeOrdinaryText(value))
            }
        }

        if addSpecialTokens, addEosToken, let eosTokenId {
            tokenIDs.append(eosTokenId)
        }
        return tokenIDs
    }

    internal func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        var data = Data()
        for token in tokens {
            if specialTokensByID[token] != nil, skipSpecialTokens {
                continue
            }
            if let byte = byteValuesByTokenID[token] {
                data.append(byte)
                continue
            }
            guard entries.indices.contains(token) else {
                continue
            }
            data.append(contentsOf: entries[token].token.utf8)
        }
        return String(decoding: data, as: UTF8.self)
    }

    internal func convertTokenToId(_ token: String) -> Int? {
        specialTokens[token] ?? tokenIDsByString[token]
    }

    internal func convertIdToToken(_ id: Int) -> String? {
        guard entries.indices.contains(id) else {
            return nil
        }
        return entries[id].token
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
        throw Tokenizers.TokenizerError.missingChatTemplate
    }

    private func insert(_ token: String, tokenID: Int, score: Int) {
        let scalars = Array(token.unicodeScalars)
        guard !scalars.isEmpty else { return }

        var node = root
        for scalar in scalars {
            if node.children[scalar] == nil {
                node.children[scalar] = TrieNode()
            }
            node = node.children[scalar]!
        }
        node.match = TrieMatch(tokenID: tokenID, score: score, length: scalars.count)
    }

    private func encodeOrdinaryText(_ text: String) -> [Int] {
        let scalars = Array(text.unicodeScalars)
        guard !scalars.isEmpty else { return [] }

        var costs = Array(repeating: Int.max / 4, count: scalars.count + 1)
        var steps = Array<EncodingStep?>(repeating: nil, count: scalars.count)
        costs[scalars.count] = 0

        for index in stride(from: scalars.count - 1, through: 0, by: -1) {
            updateBestVocabularyStep(
                from: index,
                scalars: scalars,
                costs: &costs,
                steps: &steps
            )
            updateBestByteFallbackStep(
                from: index,
                scalar: scalars[index],
                costs: &costs,
                steps: &steps
            )
        }

        var result: [Int] = []
        var index = 0
        while index < scalars.count {
            guard let step = steps[index] else {
                if let unknownTokenId {
                    result.append(unknownTokenId)
                }
                index += 1
                continue
            }
            result.append(contentsOf: step.tokenIDs)
            index = step.nextIndex
        }
        return result
    }

    private func updateBestVocabularyStep(
        from index: Int,
        scalars: [UnicodeScalar],
        costs: inout [Int],
        steps: inout [EncodingStep?]
    ) {
        var node = root
        var cursor = index
        while cursor < scalars.count, let next = node.children[scalars[cursor]] {
            cursor += 1
            node = next
            guard let match = node.match else {
                continue
            }
            let cost = costs[cursor] - match.score
            let candidate = EncodingStep(
                nextIndex: cursor,
                tokenIDs: [match.tokenID],
                length: match.length
            )
            if isBetter(cost: cost, step: candidate, than: costs[index], steps[index]) {
                costs[index] = cost
                steps[index] = candidate
            }
        }
    }

    private func updateBestByteFallbackStep(
        from index: Int,
        scalar: UnicodeScalar,
        costs: inout [Int],
        steps: inout [EncodingStep?]
    ) {
        let ids = String(scalar).utf8.compactMap { byteTokenIDsByByte[$0] }
        guard ids.count == String(scalar).utf8.count else {
            return
        }
        let cost = costs[index + 1] + 10_000_000
        let candidate = EncodingStep(nextIndex: index + 1, tokenIDs: ids, length: 1)
        if isBetter(cost: cost, step: candidate, than: costs[index], steps[index]) {
            costs[index] = cost
            steps[index] = candidate
        }
    }

    private func isBetter(
        cost: Int,
        step: EncodingStep,
        than existingCost: Int,
        _ existingStep: EncodingStep?
    ) -> Bool {
        if cost != existingCost {
            return cost < existingCost
        }
        return step.length > (existingStep?.length ?? 0)
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

    private static func loadEntries(from vocabURL: URL) throws -> [VocabularyEntry] {
        let contents = try String(contentsOf: vocabURL, encoding: .utf8)
        return try contents
            .split(whereSeparator: \.isNewline)
            .map { try parseEntry(String($0)) }
    }

    private static func parseEntry(_ line: String) throws -> VocabularyEntry {
        let data = Data(line.utf8)
        guard let row = try JSONSerialization.jsonObject(with: data) as? [Any],
              row.count >= 3,
              let token = row[0] as? String,
              let score = row[1] as? NSNumber,
              let kind = row[2] as? String
        else {
            throw Tokenizers.TokenizerError.malformedVocab
        }
        return VocabularyEntry(
            token: token,
            score: Int((score.doubleValue * 10_000).rounded()),
            kind: kind
        )
    }

    private static func byteTokenIDsByByte(_ entries: [VocabularyEntry]) -> [UInt8: Int] {
        var result: [UInt8: Int] = [:]
        result.reserveCapacity(256)
        for (index, entry) in entries.enumerated() where entry.kind == "BYTE" {
            guard let byte = byteValue(from: entry.token)
            else {
                continue
            }
            result[byte] = index
        }
        return result
    }

    private static func decodedVocabularyToken(_ entry: VocabularyEntry) -> String {
        guard entry.kind == "BYTE", let byte = byteValue(from: entry.token) else {
            return entry.token
        }
        return String(decoding: [byte], as: UTF8.self)
    }

    private static func byteValue(from token: String) -> UInt8? {
        guard token.count == 6,
              token.hasPrefix("<0x"),
              token.hasSuffix(">")
        else {
            return nil
        }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }

    private static func makeSpecialTokens(from ids: [String: Int]) -> [String: Int] {
        [
            "<|plamo:unk|>",
            "<|plamo:bos|>",
            "<|plamo:eos|>",
            "<|plamo:pad|>"
        ].reduce(into: [String: Int]()) { result, token in
            if let id = ids[token] {
                result[token] = id
            }
        }
    }
}

private struct PlamoTokenizerConfig {
    let addBosToken: Bool
    let addEosToken: Bool

    static func load(from url: URL) -> PlamoTokenizerConfig {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return PlamoTokenizerConfig(addBosToken: true, addEosToken: false)
        }

        return PlamoTokenizerConfig(
            addBosToken: object["add_bos_token"] as? Bool ?? true,
            addEosToken: object["add_eos_token"] as? Bool ?? false
        )
    }
}
