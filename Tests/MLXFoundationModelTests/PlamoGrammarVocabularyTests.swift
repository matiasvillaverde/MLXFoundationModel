import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Plamo grammar vocabulary support")
struct PlamoGrammarVocabularyTests {
    @Test("Plamo JSONL vocabularies expose byte tokens to grammar masks")
    func plamoJSONLVocabularyExposesByteTokensToGrammarMasks() async throws {
        let directory = try Self.makePlamoTokenizerDirectory()
        let exclamationID = 4 + Int(UInt8(ascii: "!"))

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            let compiler = try GrammarConstraintCompiler(modelDirectory: directory, stopTokenIds: [2])
            let matcher = try compiler.makeMatcher(
                for: GrammarSamplingConfiguration(grammar: #"root ::= "!""#)
            )
            let mask = try #require(try matcher.nextMask())

            #expect(Self.isAllowed(tokenID: exclamationID, by: mask))
            #expect(!Self.isAllowed(tokenID: 260, by: mask))
        }
        let events = Self.grammarSnapshots(from: recorded.events)

        #expect(events.contains { event in
            event.stage == MLXGrammarConstraintSnapshot.Stage.compilerReady && event.vocabularySize == 261
        })
    }

    private static func makePlamoTokenizerDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try plamoJSONLFixture().write(
            to: directory.appendingPathComponent(PlamoTokenizer.vocabFilename),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }

    private static func plamoJSONLFixture() throws -> String {
        var rows: [[Any]] = [
            ["<|plamo:unk|>", 0.0, "UNKNOWN"],
            ["<|plamo:bos|>", 0.0, "CONTROL"],
            ["<|plamo:eos|>", 0.0, "CONTROL"],
            ["<|plamo:pad|>", 0.0, "CONTROL"]
        ]
        rows.append(contentsOf: (0 ... 255).map { byte in
            [String(format: "<0x%02X>", byte), 0.0, "BYTE"] as [Any]
        })
        rows.append(["hello", 2.0, "NORMAL"])

        return try rows.map { row in
            let data = try JSONSerialization.data(withJSONObject: row)
            guard let line = String(data: data, encoding: .utf8) else {
                throw FixtureError.invalidJSONLine
            }
            return line
        }
        .joined(separator: "\n")
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

    private enum FixtureError: Error {
        case invalidJSONLine
    }
}
