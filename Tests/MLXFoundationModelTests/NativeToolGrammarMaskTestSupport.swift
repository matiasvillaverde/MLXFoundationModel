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
        let prefixTokens = prefixTokens(for: prefix, grammar: grammar)
        let allowedTokenID = prefixTokens.count + 1
        let rejectedTokenID = prefixTokens.count + 2
        let matcher = try matcher(
            vocabulary: ["</s>"] + prefixTokens + [allowed, rejected],
            grammar: grammar
        )
        if !prefixTokens.isEmpty {
            for tokenID in 1 ... prefixTokens.count {
                try matcher.accept(token: Int32(tokenID))
            }
        }
        let mask = try #require(try matcher.nextMask())

        #expect(isAllowed(tokenID: allowedTokenID, by: mask))
        #expect(!isAllowed(tokenID: rejectedTokenID, by: mask))
    }

    private static func prefixTokens(
        for prefix: String,
        grammar: GrammarSamplingConfiguration
    ) -> [String] {
        guard grammar.kind == .structuralTag else {
            return [prefix]
        }
        for marker in [
            "<｜DSML｜parameter",
            "<parameter=",
            "<arg_key>",
            "<parameter name=\""
        ] {
            if let range = prefix.range(of: marker, options: .backwards) {
                return [
                    String(prefix[..<range.lowerBound]),
                    String(prefix[range.lowerBound...])
                ].filter { !$0.isEmpty }
            }
        }
        return [prefix]
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
