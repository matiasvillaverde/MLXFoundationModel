import Foundation
import Tokenizers

internal final class SentencePieceModelTokenizer: @unchecked Sendable, Tokenizer {
    internal static let modelFilename = "tokenizer.model"

    private enum PieceType: Int, Sendable {
        case normal = 1
        case unknown = 2
        case control = 3
        case userDefined = 4
        case unused = 5
        case byte = 6
    }

    private struct Piece: Sendable {
        let token: String
        let score: Float
        let type: PieceType

        var isStructuralSpecialToken: Bool {
            type == .unknown || type == .control || type == .unused
        }
    }

    fileprivate struct TokenizerConfig {
        let chatTemplate: String?
        let addBosToken: Bool
        let addEosToken: Bool
        let bosToken: String?
        let eosToken: String?
        let unknownToken: String?
        let specialTokens: [String: Int]
    }

    private let pieces: [Piece]
    private let tokenIDsByPiece: [String: Int]
    private let textTokenIDsByPiece: [String: Int]
    private let specialTokens: [String: Int]
    private let specialTokensByID: [Int: String]
    private let specialTokensByFirstCharacter: [Character: [String]]
    private let addBosToken: Bool
    private let addEosToken: Bool
    private let chatTemplate: String?

    internal let bosToken: String?
    internal let bosTokenId: Int?
    internal let eosToken: String?
    internal let eosTokenId: Int?
    internal let unknownToken: String?
    internal let unknownTokenId: Int?
    internal var fuseUnknownTokens: Bool { true }
    internal var hasChatTemplate: Bool { chatTemplate != nil }

    internal static func load(from directory: URL) throws -> SentencePieceModelTokenizer? {
        let modelURL = directory.appending(component: modelFilename)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            return nil
        }

        let config = try TokenizerConfig.load(
            from: directory.appending(component: "tokenizer_config.json")
        )
        return try SentencePieceModelTokenizer(modelURL: modelURL, config: config)
    }

    internal static func vocabularyEntries(from modelURL: URL, configURL: URL?) throws -> [Int: String] {
        let config: TokenizerConfig
        if let configURL {
            config = try TokenizerConfig.load(from: configURL)
        } else {
            config = .empty
        }
        let tokenizer = try SentencePieceModelTokenizer(modelURL: modelURL, config: config)
        return tokenizer.decodedVocabularyEntries()
    }

    private init(modelURL: URL, config: TokenizerConfig) throws {
        let pieces = try Self.loadPieces(from: modelURL)
        let tokenIDsByPiece = Self.makeTokenIDMap(from: pieces)
        let textTokenIDsByPiece = Self.makeTokenIDMap(from: pieces) { piece in
            piece.type == .normal || piece.type == .userDefined || piece.type == .byte
        }

        var specialTokens = config.specialTokens
        for (index, piece) in pieces.enumerated() where piece.isStructuralSpecialToken {
            specialTokens[piece.token] = index
        }
        let specialTokensByID = Self.makeReverseTokenMap(from: specialTokens)
        let orderedSpecialTokens = specialTokens.keys.filter { !$0.isEmpty }.sorted { lhs, rhs in
            if lhs.count != rhs.count { return lhs.count > rhs.count }
            return lhs < rhs
        }

        self.pieces = pieces
        self.tokenIDsByPiece = tokenIDsByPiece
        self.textTokenIDsByPiece = textTokenIDsByPiece
        self.specialTokens = specialTokens
        self.specialTokensByID = specialTokensByID
        self.specialTokensByFirstCharacter = Dictionary(
            grouping: orderedSpecialTokens,
            by: { $0.first! }
        )
        self.addBosToken = config.addBosToken
        self.addEosToken = config.addEosToken
        self.chatTemplate = config.chatTemplate
        self.bosToken = config.bosToken
        self.eosToken = config.eosToken
        self.unknownToken = config.unknownToken
        self.bosTokenId = config.bosToken.flatMap { tokenIDsByPiece[$0] ?? specialTokens[$0] }
        self.eosTokenId = config.eosToken.flatMap { tokenIDsByPiece[$0] ?? specialTokens[$0] }
        self.unknownTokenId = config.unknownToken.flatMap { tokenIDsByPiece[$0] ?? specialTokens[$0] }
    }

    internal func tokenize(text: String) -> [String] {
        encode(text: text, addSpecialTokens: false).compactMap(convertIdToToken)
    }

    internal func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    internal func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens: [Int] = []
        if addSpecialTokens, addBosToken, let bosTokenId {
            tokens.append(bosTokenId)
        }
        for segment in splitSpecialSegments(text) {
            switch segment {
            case .special(let token):
                if let id = specialTokens[token] ?? tokenIDsByPiece[token] {
                    tokens.append(id)
                }
            case .text(let value):
                appendText(value, to: &tokens)
            }
        }
        if addSpecialTokens, addEosToken, let eosTokenId {
            tokens.append(eosTokenId)
        }
        return tokens
    }

    internal func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        var text = ""
        var pendingBytes = Data()

        func flushBytes() {
            guard !pendingBytes.isEmpty else { return }
            text += String(decoding: pendingBytes, as: UTF8.self)
            pendingBytes.removeAll(keepingCapacity: true)
        }

        for token in tokens {
            guard pieces.indices.contains(token) else { continue }
            let piece = pieces[token]

            if piece.type == .byte, let byte = Self.byteValue(from: piece.token) {
                pendingBytes.append(byte)
                continue
            }

            flushBytes()

            if specialTokensByID[token] != nil || piece.type == .control || piece.type == .unknown {
                if !skipSpecialTokens {
                    text += piece.token
                }
                continue
            }

            text += Self.decodePiece(piece.token)
        }

        flushBytes()
        if text.first == " " {
            text.removeFirst()
        }
        return text
    }

    internal func convertTokenToId(_ token: String) -> Int? {
        specialTokens[token] ?? tokenIDsByPiece[token]
    }

    internal func convertIdToToken(_ id: Int) -> String? {
        pieces.indices.contains(id) ? pieces[id].token : specialTokensByID[id]
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
                "SentencePiece model tokenizer does not define a tool-use template"
            )
        }
        if additionalContext?.isEmpty == false {
            throw Tokenizers.TokenizerError.chatTemplate(
                "SentencePiece model tokenizer does not use additional template context"
            )
        }
        try validate(chatTemplate)

        let rendered = try render(messages: messages, addGenerationPrompt: addGenerationPrompt)
        var encoded = encode(text: rendered, addSpecialTokens: false)
        if let maxLength, encoded.count > maxLength, truncation {
            encoded = Array(encoded.prefix(maxLength))
        }
        return encoded
    }

    private func appendText(_ text: String, to tokens: inout [Int]) {
        let normalized = Self.normalize(text)
        var cursor = normalized.startIndex
        while cursor < normalized.endIndex {
            if let match = longestPiece(in: normalized, at: cursor) {
                tokens.append(match.id)
                cursor = match.end
                continue
            }

            let scalarText = String(normalized[cursor])
            for byte in scalarText.utf8 {
                let bytePiece = String(format: "<0x%02X>", byte)
                if let id = tokenIDsByPiece[bytePiece] {
                    tokens.append(id)
                } else if let unknownTokenId {
                    tokens.append(unknownTokenId)
                }
            }
            cursor = normalized.index(after: cursor)
        }
    }

    private func longestPiece(in text: String, at start: String.Index) -> (id: Int, end: String.Index)? {
        var end = text.index(after: start)
        var best: (id: Int, end: String.Index)?

        while true {
            let candidate = String(text[start ..< end])
            if let id = textTokenIDsByPiece[candidate] {
                best = (id, end)
            }
            guard end < text.endIndex else { break }
            end = text.index(after: end)
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

    private func validate(_ argument: ChatTemplateArgument?) throws {
        guard let argument else { return }
        switch argument {
        case .literal(let template):
            guard template == chatTemplate else {
                throw Tokenizers.TokenizerError.chatTemplate(
                    "SentencePiece model tokenizer only supports its configured chat template"
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
        guard chatTemplate != nil else {
            return messages.compactMap { Self.stringContent(from: $0["content"]) }
                .joined(separator: "\n")
        }

        var result = bosToken ?? ""
        for message in messages {
            guard let role = message["role"] as? String else {
                throw Tokenizers.TokenizerError.chatTemplate("Message is missing a string role")
            }
            let content = Self.stringContent(from: message["content"])
            result += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        if addGenerationPrompt {
            result += "<|im_start|>assistant\n"
        }
        return result
    }

    private func decodedVocabularyEntries() -> [Int: String] {
        pieces.enumerated().reduce(into: [Int: String]()) { result, element in
            let (id, piece) = element
            if specialTokensByID[id] != nil || piece.type == .control || piece.type == .unknown {
                result[id] = piece.token
            } else {
                result[id] = Self.decodePiece(piece.token)
            }
        }
    }

    private static func makeTokenIDMap(
        from pieces: [Piece],
        include: (Piece) -> Bool = { _ in true }
    ) -> [String: Int] {
        pieces.enumerated().reduce(into: [String: Int]()) { result, element in
            let (id, piece) = element
            guard include(piece), result[piece.token] == nil else { return }
            result[piece.token] = id
        }
    }

    private static func makeReverseTokenMap(from tokens: [String: Int]) -> [Int: String] {
        tokens.reduce(into: [Int: String]()) { result, element in
            let (token, id) = element
            guard result[id] == nil else { return }
            result[id] = token
        }
    }

    private static func loadPieces(from url: URL) throws -> [Piece] {
        let data = try Data(contentsOf: url)
        var reader = ProtobufReader(data)
        var pieces: [Piece] = []

        while let field = try reader.nextField() {
            guard field.number == 1, case .bytes(let pieceData) = field.value else {
                continue
            }
            pieces.append(try parsePiece(pieceData))
        }
        return pieces
    }

    private static func parsePiece(_ data: Data) throws -> Piece {
        var reader = ProtobufReader(data)
        var token: String?
        var score: Float = 0
        var type = PieceType.normal

        while let field = try reader.nextField() {
            switch (field.number, field.value) {
            case (1, .bytes(let value)):
                token = String(data: value, encoding: .utf8)
            case (2, .fixed32(let bitPattern)):
                score = Float(bitPattern: bitPattern)
            case (3, .varint(let rawType)):
                type = PieceType(rawValue: Int(rawType)) ?? .normal
            default:
                continue
            }
        }

        guard let token else {
            throw Tokenizers.TokenizerError.malformedVocab
        }
        return Piece(token: token, score: score, type: type)
    }

    private static func normalize(_ text: String) -> String {
        let escaped = text.replacingOccurrences(of: " ", with: "▁")
        guard !escaped.isEmpty else {
            return escaped
        }
        return escaped.hasPrefix("▁") ? escaped : "▁\(escaped)"
    }

    private static func decodePiece(_ token: String) -> String {
        token.replacingOccurrences(of: "▁", with: " ")
    }

    private static func byteValue(from token: String) -> UInt8? {
        guard token.hasPrefix("<0x"), token.hasSuffix(">") else {
            return nil
        }
        let start = token.index(token.startIndex, offsetBy: 3)
        let end = token.index(before: token.endIndex)
        return UInt8(token[start ..< end], radix: 16)
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

private extension SentencePieceModelTokenizer.TokenizerConfig {
    static let empty = Self(
        chatTemplate: nil,
        addBosToken: true,
        addEosToken: false,
        bosToken: nil,
        eosToken: nil,
        unknownToken: nil,
        specialTokens: [:]
    )

    static func load(from url: URL) throws -> Self {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .empty
        }
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .empty
        }

        var specialTokens: [String: Int] = [:]
        if let decoder = object["added_tokens_decoder"] as? [String: Any] {
            for (key, rawValue) in decoder {
                guard let id = Int(key),
                      let token = (rawValue as? [String: Any])?["content"] as? String else {
                    continue
                }
                specialTokens[token] = id
            }
        }

        return Self(
            chatTemplate: object["chat_template"] as? String,
            addBosToken: object["add_bos_token"] as? Bool ?? true,
            addEosToken: object["add_eos_token"] as? Bool ?? false,
            bosToken: tokenString(object["bos_token"]),
            eosToken: tokenString(object["eos_token"]),
            unknownToken: tokenString(object["unk_token"]),
            specialTokens: specialTokens
        )
    }

    private static func tokenString(_ value: Any?) -> String? {
        if let string = value as? String {
            return string
        }
        return (value as? [String: Any])?["content"] as? String
    }
}

private struct ProtobufReader {
    enum Value {
        case varint(UInt64)
        case bytes(Data)
        case fixed32(UInt32)
        case fixed64(UInt64)
    }

    struct Field {
        let number: Int
        let value: Value
    }

    private let bytes: [UInt8]
    private var index: Int = 0

    init(_ data: Data) {
        self.bytes = Array(data)
    }

    mutating func nextField() throws -> Field? {
        guard index < bytes.count else {
            return nil
        }

        let key = try readVarint()
        let fieldNumber = Int(key >> 3)
        let wireType = Int(key & 0x7)

        switch wireType {
        case 0:
            return Field(number: fieldNumber, value: .varint(try readVarint()))
        case 1:
            return Field(number: fieldNumber, value: .fixed64(try readFixed64()))
        case 2:
            let count = Int(try readVarint())
            guard index + count <= bytes.count else {
                throw Tokenizers.TokenizerError.malformedVocab
            }
            let value = Data(bytes[index ..< index + count])
            index += count
            return Field(number: fieldNumber, value: .bytes(value))
        case 5:
            return Field(number: fieldNumber, value: .fixed32(try readFixed32()))
        default:
            throw Tokenizers.TokenizerError.malformedVocab
        }
    }

    private mutating func readVarint() throws -> UInt64 {
        var value: UInt64 = 0
        var shift: UInt64 = 0

        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            value |= UInt64(byte & 0x7F) << shift
            if byte < 0x80 {
                return value
            }
            shift += 7
            if shift >= 64 {
                throw Tokenizers.TokenizerError.malformedVocab
            }
        }
        throw Tokenizers.TokenizerError.malformedVocab
    }

    private mutating func readFixed32() throws -> UInt32 {
        guard index + 4 <= bytes.count else {
            throw Tokenizers.TokenizerError.malformedVocab
        }
        defer { index += 4 }
        return UInt32(bytes[index])
            | (UInt32(bytes[index + 1]) << 8)
            | (UInt32(bytes[index + 2]) << 16)
            | (UInt32(bytes[index + 3]) << 24)
    }

    private mutating func readFixed64() throws -> UInt64 {
        guard index + 8 <= bytes.count else {
            throw Tokenizers.TokenizerError.malformedVocab
        }
        defer { index += 8 }
        return (0 ..< 8).reduce(UInt64(0)) { value, offset in
            value | (UInt64(bytes[index + offset]) << UInt64(8 * offset))
        }
    }
}
