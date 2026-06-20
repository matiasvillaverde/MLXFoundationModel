import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Grammar constrained decoding")
struct GrammarConstrainedDecodingTests {
    @Test("EBNF constraints mask invalid high-logit tokens")
    func ebnfConstraintsMaskInvalidTokens() throws {
        let matcher = try Self.matcher(
            vocabulary: ["</s>", "a", "b", "c"],
            grammar: GrammarSamplingConfiguration(grammar: #"root ::= "b""#)
        )
        let nextMask = try matcher.nextMask()
        let mask = try #require(nextMask)

        #expect(!Self.isAllowed(tokenID: 1, by: mask))
        #expect(Self.isAllowed(tokenID: 2, by: mask))
        #expect(!Self.isAllowed(tokenID: 3, by: mask))
    }

    @Test("EBNF constraints advance after accepted samples")
    func ebnfConstraintsAdvanceAfterAcceptedSamples() throws {
        let matcher = try Self.matcher(
            vocabulary: ["</s>", "a", "b", "c"],
            grammar: GrammarSamplingConfiguration(grammar: #"root ::= "a" "b""#)
        )
        let nextFirst = try matcher.nextMask()
        let first = try #require(nextFirst)

        try matcher.accept(token: 1)
        let nextSecond = try matcher.nextMask()
        let second = try #require(nextSecond)

        #expect(Self.isAllowed(tokenID: 1, by: first))
        #expect(!Self.isAllowed(tokenID: 3, by: first))
        #expect(Self.isAllowed(tokenID: 2, by: second))
        #expect(!Self.isAllowed(tokenID: 3, by: second))
    }

    @Test("finite choices mask every non-choice token")
    func finiteChoicesMaskEveryNonChoiceToken() throws {
        let vocabulary = ["</s>", "apple", "pear", "banana", "orange"]
        let matcher = try Self.matcher(
            vocabulary: vocabulary,
            grammar: .choices(["apple", "pear", "banana"])
        )
        let nextMask = try matcher.nextMask()
        let mask = try #require(nextMask)

        #expect(Self.isAllowed(tokenID: 1, by: mask))
        #expect(Self.isAllowed(tokenID: 2, by: mask))
        #expect(Self.isAllowed(tokenID: 3, by: mask))
        #expect(!Self.isAllowed(tokenID: 4, by: mask))
    }

    @Test("finite choice diagnostics use choice kind")
    func finiteChoiceDiagnosticsUseChoiceKind() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let matcher = try Self.matcher(
                vocabulary: ["</s>", "apple", "pear", "banana", "orange"],
                grammar: .choices(["apple", "pear", "banana"])
            )
            _ = try matcher.nextMask()
            try matcher.accept(token: 1)
        }
        let events = Self.grammarSnapshots(from: recorded.events)

        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == .choices })
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == .choices })
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == .choices && $0.tokenID == 1 })
    }

    @Test("JSON schema constraints force a JSON object start")
    func jsonSchemaConstraintsForceJSONObjectStart() throws {
        let matcher = try Self.matcher(
            vocabulary: Self.jsonVocabulary,
            grammar: .jsonSchema(Self.weatherSchema)
        )
        let openBraceID = try #require(Self.jsonVocabulary.firstIndex(of: "{"))
        let badID = try #require(Self.jsonVocabulary.firstIndex(of: "bad"))
        let nextMask = try matcher.nextMask()
        let mask = try #require(nextMask)

        #expect(Self.isAllowed(tokenID: openBraceID, by: mask))
        #expect(!Self.isAllowed(tokenID: badID, by: mask))
    }

    @Test("JSON schema string enums constrain choice tokens")
    func jsonSchemaStringEnumsConstrainChoiceTokens() throws {
        let vocabulary = ["</s>", "\"", "apple", "pear", "banana", "orange"]
        let matcher = try Self.matcher(
            vocabulary: vocabulary,
            grammar: .jsonSchema(Self.fruitSchema)
        )
        let first = try #require(try matcher.nextMask())

        try matcher.accept(token: 1)
        let choice = try #require(try matcher.nextMask())

        #expect(Self.isAllowed(tokenID: 1, by: first))
        #expect(!Self.isAllowed(tokenID: 5, by: first))
        #expect(Self.isAllowed(tokenID: 2, by: choice))
        #expect(Self.isAllowed(tokenID: 3, by: choice))
        #expect(Self.isAllowed(tokenID: 4, by: choice))
        #expect(!Self.isAllowed(tokenID: 5, by: choice))
    }

    @Test("BPE vocab parser preserves Unicode-distinct token keys")
    func bpeVocabParserPreservesUnicodeDistinctTokenKeys() throws {
        let matcher = try Self.matcher(
            vocabulary: ["</s>", "\u{03AD}", "\u{03B5}\u{0301}", "{"],
            grammar: GrammarSamplingConfiguration(grammar: #"root ::= "{""#)
        )
        let nextMask = try matcher.nextMask()
        let mask = try #require(nextMask)

        #expect(Self.isAllowed(tokenID: 3, by: mask))
    }

    @Test("streaming vocab parser handles array vocabularies and added tokens")
    func streamingVocabParserHandlesArrayVocabulariesAndAddedTokens() async throws {
        let tokenizer = """
        {"model":{"type":"Unigram","vocab":[["</s>",0],["apple",-1]]},\
        "added_tokens":[{"id":2,"content":"banana","single_word":false}]}
        """
        let directory = try Self.makeTokenizerDirectory(tokenizerJSON: tokenizer)
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let compiler = try GrammarConstraintCompiler(modelDirectory: directory, stopTokenIds: [0])
            let matcher = try compiler.makeMatcher(for: .choices(["banana"]))
            let mask = try #require(try matcher.nextMask())
            #expect(Self.isAllowed(tokenID: 2, by: mask))
        }
        let events = Self.grammarSnapshots(from: recorded.events)

        #expect(events.contains { event in
            event.stage == MLXGrammarConstraintSnapshot.Stage.compilerReady && event.vocabularySize == 3
        })
    }

    @Test("grammar matcher records compiler, mask, and accepted-token diagnostics")
    func grammarMatcherRecordsDiagnostics() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let matcher = try Self.matcher(
                vocabulary: ["</s>", "a", "b", "c"],
                grammar: GrammarSamplingConfiguration(grammar: #"root ::= "b""#)
            )
            _ = try matcher.nextMask()
            try matcher.accept(token: 2)
        }
        let events = Self.grammarSnapshots(from: recorded.events)

        #expect(events.contains { $0.stage == .compilerReady && $0.vocabularySize == 4 })
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == .ebnf })
        #expect(events.contains { $0.stage == .maskApplied && $0.mode == .allow && $0.tokenCount == 1 })
        #expect(events.contains { $0.stage == .tokenAccepted && $0.tokenID == 2 })
    }

    @Test("grammar matcher records rejected tokens")
    func grammarMatcherRecordsRejectedTokens() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let matcher = try Self.matcher(
                vocabulary: ["</s>", "a", "b", "c"],
                grammar: GrammarSamplingConfiguration(grammar: #"root ::= "b""#)
            )
            do {
                try matcher.accept(token: 1)
                Issue.record("Expected matcher to reject token 1")
            } catch GrammarConstraintError.rejectedSampledToken {
                return
            }
        }
        let events = Self.grammarSnapshots(from: recorded.events)

        #expect(events.contains { $0.stage == .tokenRejected && $0.tokenID == 1 })
    }

    private static func matcher(
        vocabulary: [String],
        grammar: GrammarSamplingConfiguration
    ) throws -> GrammarConstraintMatcher {
        try compiler(vocabulary: vocabulary).makeMatcher(for: grammar)
    }

    private static func compiler(vocabulary: [String]) throws -> GrammarConstraintCompiler {
        let directory = try makeTokenizerDirectory(vocabulary: vocabulary)
        return try GrammarConstraintCompiler(modelDirectory: directory, stopTokenIds: [0])
    }

    private static func isAllowed(tokenID: Int, by mask: GrammarTokenMask) -> Bool {
        let containsToken = mask.tokenIDs.contains(Int32(tokenID))
        switch mask.mode {
        case .allow:
            return containsToken

        case .reject:
            return !containsToken
        }
    }

    private static func allowedTokenIDs(
        in vocabulary: [String],
        by mask: GrammarTokenMask
    ) -> [Int] {
        vocabulary.indices.filter { tokenID in
            isAllowed(tokenID: tokenID, by: mask)
        }
    }

    private static func grammarSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXGrammarConstraintSnapshot] {
        events.compactMap { event in
            guard case .grammarConstraint(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func makeTokenizerDirectory(vocabulary: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tokenizer = """
        {"model":{"type":"BPE","vocab":\(Self.vocabJSON(vocabulary))},"decoder":{"type":"Raw"}}
        """
        return try makeTokenizerDirectory(tokenizerJSON: tokenizer)
    }

    private static func makeTokenizerDirectory(tokenizerJSON: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        try tokenizerJSON.write(to: tokenizerURL, atomically: true, encoding: .utf8)
        return directory
    }

    private static func vocabJSON(_ vocabulary: [String]) -> String {
        let entries = vocabulary.enumerated().map { index, token in
            #""\#(escaped(token))":\#(index)"#
        }
        return "{\(entries.joined(separator: ","))}"
    }

    private static func escaped(_ token: String) -> String {
        token
            .replacingOccurrences(of: #"\"#, with: #"\\"#)
            .replacingOccurrences(of: #"""#, with: #"\""#)
            .replacingOccurrences(of: "\n", with: #"\n"#)
    }

    private static let jsonVocabulary = [
        "</s>", "{", "}", "bad", "\"", "city", "celsius", "Berlin", "21", ":", ",", " ", "\n"
    ]

    private static let weatherSchema = """
    {"type":"object","properties":{"city":{"enum":["Berlin"]},"celsius":{"enum":[21]}},\
    "required":["city","celsius"],"additionalProperties":false}
    """

    private static let fruitSchema = #"{"enum":["apple","pear","banana"]}"#
}
