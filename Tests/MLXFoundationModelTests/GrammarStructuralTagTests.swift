import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Grammar structural tags")
struct GrammarStructuralTagTests {
    @Test("structural tags force native tool envelope starts")
    func structuralTagsForceNativeToolEnvelopeStarts() throws {
        let matcher = try Self.matcher(
            vocabulary: [
                "</s>",
                "{",
                "<tool_call><function=weather>"
            ],
            grammar: .structuralTag(Self.qwenStructuralTag)
        )
        let mask = try #require(try matcher.nextMask())

        #expect(Self.isAllowed(tokenID: 2, by: mask))
        #expect(!Self.isAllowed(tokenID: 1, by: mask))
    }

    @Test("structural tags constrain XML parameter values from schema")
    func structuralTagsConstrainXMLParameterValuesFromSchema() throws {
        let matcher = try Self.matcher(
            vocabulary: [
                "</s>",
                "<tool_call><function=weather>",
                "<parameter=count>",
                "2",
                "Berlin"
            ],
            grammar: .structuralTag(Self.qwenStructuralTag)
        )

        try matcher.accept(token: 1)
        let parameterMask = try #require(try matcher.nextMask())
        try matcher.accept(token: 2)
        let valueMask = try #require(try matcher.nextMask())

        #expect(Self.isAllowed(tokenID: 2, by: parameterMask))
        #expect(!Self.isAllowed(tokenID: 4, by: parameterMask))
        #expect(Self.isAllowed(tokenID: 3, by: valueMask))
        #expect(!Self.isAllowed(tokenID: 4, by: valueMask))
    }

    @Test("DeepSeek structural tags preserve non-string XML values")
    func deepSeekStructuralTagsPreserveNonStringXMLValues() throws {
        let matcher = try Self.matcher(
            vocabulary: [
                "</s>",
                Self.deepSeekBegin,
                #"<｜DSML｜parameter name="count" string="false">"#,
                "2",
                "Berlin"
            ],
            grammar: .structuralTag(Self.deepSeekStructuralTag)
        )

        try matcher.accept(token: 1)
        let parameterMask = try #require(try matcher.nextMask())
        try matcher.accept(token: 2)
        let valueMask = try #require(try matcher.nextMask())

        #expect(Self.isAllowed(tokenID: 2, by: parameterMask))
        #expect(!Self.isAllowed(tokenID: 4, by: parameterMask))
        #expect(Self.isAllowed(tokenID: 3, by: valueMask))
        #expect(!Self.isAllowed(tokenID: 4, by: valueMask))
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

    private static let qwenStructuralTag = """
    {"type":"structural_tag","format":{"type":"tag","begin":"<tool_call><function=weather>",\
    "content":{"type":"json_schema","style":"qwen_xml","json_schema":\(weatherCountSchema)},\
    "end":"</function></tool_call>"}}
    """

    private static let deepSeekBegin = "\n\n<｜DSML｜tool_calls>\n"
        + #"<｜DSML｜invoke name="weather">"#
        + "\n"

    private static let deepSeekStructuralTag = """
    {"type":"structural_tag","format":{"type":"tag","begin":\(jsonString(deepSeekBegin)),\
    "content":{"type":"json_schema","style":"deepseek_xml","json_schema":\(weatherCountSchema)},\
    "end":"</｜DSML｜invoke>\\n</｜DSML｜tool_calls>"}}
    """

    private static let weatherCountSchema = """
    {"type":"object","properties":{"count":{"type":"integer"}},\
    "required":["count"],"additionalProperties":false}
    """

    private static func jsonString(_ value: String) -> String {
        #""\#(escaped(value))""#
    }
}
