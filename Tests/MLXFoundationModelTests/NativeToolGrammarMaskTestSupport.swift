import Foundation
@testable import MLXLocalModels
import Testing

enum NativeToolGrammarMaskTestSupport {
    static func expectRejectsJSONStart(
        _ grammar: GrammarSamplingConfiguration,
        nativeStart: String = "<tool_call><function=weather>"
    ) throws {
        let matcher = try matcher(
            vocabulary: ["</s>", "{", nativeStart],
            grammar: grammar
        )
        let mask = try #require(try matcher.nextMask())

        #expect(isAllowed(tokenID: 2, by: mask))
        #expect(!isAllowed(tokenID: 1, by: mask))
    }

    static func expectAllowsOnlyAfterAcceptingPrefix(
        _ grammar: GrammarSamplingConfiguration,
        prefix: String,
        allowed: String,
        rejected: String
    ) throws {
        let matcher = try matcher(
            vocabulary: ["</s>", prefix, allowed, rejected],
            grammar: grammar
        )
        try matcher.accept(token: 1)
        let mask = try #require(try matcher.nextMask())

        #expect(isAllowed(tokenID: 2, by: mask))
        #expect(!isAllowed(tokenID: 3, by: mask))
    }

    private static func matcher(
        vocabulary: [String],
        grammar: GrammarSamplingConfiguration
    ) throws -> GrammarConstraintMatcher {
        try GrammarConstraintCompiler(
            modelDirectory: makeTokenizerDirectory(vocabulary: vocabulary),
            stopTokenIds: [0]
        )
        .makeMatcher(for: grammar)
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

    private static func makeTokenizerDirectory(vocabulary: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try tokenizerJSON(vocabulary).write(
            to: directory.appendingPathComponent("tokenizer.json"),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }

    private static func tokenizerJSON(_ vocabulary: [String]) -> String {
        """
        {"model":{"type":"BPE","vocab":\(vocabJSON(vocabulary))},"decoder":{"type":"Raw"}}
        """
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
}
