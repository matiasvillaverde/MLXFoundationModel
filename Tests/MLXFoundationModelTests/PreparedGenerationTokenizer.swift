import Tokenizers

// swiftlint:disable discouraged_optional_collection function_parameter_count
struct PreparedGenerationTokenizer: Tokenizer {
    private let tokenIDsByText = [
        "hello": 10,
        "world": 11,
        "ST": 20,
        "OP hidden": 21,
        "<unk>": 0,
        "<eos>": 99,
        "<extra>": 100
    ]
    private let textByTokenID = [
        0: "<unk>",
        1: "A",
        2: "B",
        10: "hello",
        11: "world",
        20: "ST",
        21: "OP hidden",
        99: "<eos>",
        100: "<extra>"
    ]

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<eos>" }
    var eosTokenId: Int? { 99 }
    var unknownToken: String? { "<unk>" }
    var unknownTokenId: Int? { 0 }

    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map(String.init)
    }

    func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        tokenize(text: text).map { tokenIDsByText[$0] ?? unknownTokenId ?? 0 }
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens
            .filter { tokenID in !skipSpecialTokens || !specialTokenIDs.contains(tokenID) }
            .compactMap { textByTokenID[$0] }
            .joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenIDsByText[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        textByTokenID[id]
    }

    func applyChatTemplate(messages: [Message]) throws -> [Int] {
        encodeChatMessages(messages)
    }

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        encodeChatMessages(messages)
    }

    func applyChatTemplate(
        messages: [Message],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encodeChatMessages(messages)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument
    ) throws -> [Int] {
        encodeChatMessages(messages)
    }

    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        encodeChatMessages(messages)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int] {
        encodeChatMessages(messages)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encodeChatMessages(messages)
    }

    private var specialTokenIDs: Set<Int> {
        let candidateTokenIDs = [unknownTokenId, eosTokenId, convertTokenToId("<extra>")]
        return candidateTokenIDs.compactMap(\.self).reduce(into: []) { tokenIDs, tokenID in
            tokenIDs.insert(tokenID)
        }
    }

    private func encodeChatMessages(_ messages: [Message]) -> [Int] {
        let prompt = messages
            .compactMap { message in message["content"] as? String }
            .joined(separator: " ")
        return encode(text: prompt)
    }
}
// swiftlint:enable discouraged_optional_collection function_parameter_count
