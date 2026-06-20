import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Grammar constrained decoding batching")
struct GrammarConstrainedDecodingBatchTests {
    @Test("batched grammar masks match independent single-row masks")
    func batchedGrammarMasksMatchIndependentSingleRowMasks() async throws {
        let vocabulary = ["</s>", "a", "b", "c"]
        let configuration = GrammarSamplingConfiguration(grammar: #"root ::= "a" "b""#)
        let compiler = try Self.compiler(vocabulary: vocabulary)
        let firstBatched = try compiler.makeMatcher(for: configuration)
        let secondBatched = try compiler.makeMatcher(for: configuration)
        try secondBatched.accept(token: 1)
        let firstSingle = try compiler.makeMatcher(for: configuration)
        let secondSingle = try compiler.makeMatcher(for: configuration)
        try secondSingle.accept(token: 1)

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try GrammarConstraintMatcher.nextMasksBatched(for: [firstBatched, secondBatched])
        }
        let batched = recorded.result
        let singleFirst = try #require(try firstSingle.nextMask())
        let singleSecond = try #require(try secondSingle.nextMask())
        let events = Self.grammarSnapshots(from: recorded.events)
        let firstBatchIDs = Self.allowedTokenIDs(in: vocabulary, by: batched[0])
        let secondBatchIDs = Self.allowedTokenIDs(in: vocabulary, by: batched[1])

        #expect(firstBatchIDs == Self.allowedTokenIDs(in: vocabulary, by: singleFirst))
        #expect(secondBatchIDs == Self.allowedTokenIDs(in: vocabulary, by: singleSecond))
        #expect(events.filter { $0.stage == .batchMaskApplied }.count == 2)
    }

    @Test(
        "batched grammar logits preserve row-local masks",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func batchedGrammarLogitsPreserveRowLocalMasks() async throws {
        let vocabulary = ["</s>", "a", "b", "c"]
        let configuration = GrammarSamplingConfiguration(grammar: #"root ::= "a" "b""#)
        let compiler = try Self.compiler(vocabulary: vocabulary)

        let recorded = try await Device.withDefaultDevice(.cpu) {
            let firstMatcher = try compiler.makeMatcher(for: configuration)
            let secondMatcher = try compiler.makeMatcher(for: configuration)
            try secondMatcher.accept(token: 1)
            var processors: [GrammarConstrainedLogitProcessor?] = [
                GrammarConstrainedLogitProcessor(matcher: firstMatcher),
                GrammarConstrainedLogitProcessor(matcher: secondMatcher)
            ]
            let logits = MLXArray([
                Float(0), Float(10), Float(20), Float(30),
                Float(100), Float(110), Float(120), Float(130)
            ]).reshaped([2, 4])

            return try await MLXGenerationDiagnostics.withRecording {
                let masked = GrammarConstrainedLogitProcessor.processBatched(
                    logits: logits,
                    processors: &processors
                )
                eval(masked)
                return Self.rows(masked, columnCount: vocabulary.count)
            }
        }
        let events = Self.grammarSnapshots(from: recorded.events)

        #expect(recorded.result[0] == [-Float.infinity, 10, -Float.infinity, -Float.infinity])
        #expect(recorded.result[1] == [-Float.infinity, -Float.infinity, 120, -Float.infinity])
        #expect(events.filter { $0.stage == .batchMaskApplied }.count == 2)
        #expect(events.filter { $0.stage == .mlxMaskPrepared }.count == 2)
    }

    @Test(
        "row table grammar logits stay aligned after row removal and late join",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func rowTableGrammarLogitsStayAlignedAfterRowRemovalAndLateJoin() async throws {
        let vocabulary = ["</s>", "a", "b", "c"]
        let configuration = GrammarSamplingConfiguration(grammar: #"root ::= "a" "b""#)
        let compiler = try Self.compiler(vocabulary: vocabulary)

        let recorded = try await Device.withDefaultDevice(.cpu) {
            try await Self.rowTableGrammarResult(
                vocabulary: vocabulary,
                compiler: compiler,
                configuration: configuration
            )
        }
        let events = Self.grammarSnapshots(from: recorded.events)

        #expect(recorded.result.ids == [20, 30])
        #expect(recorded.result.rows[0] == [-Float.infinity, -Float.infinity, 120, -Float.infinity])
        #expect(recorded.result.rows[1] == [-Float.infinity, 10, -Float.infinity, -Float.infinity])
        #expect(events.filter { $0.stage == .batchMaskApplied }.count == 2)
    }

    private struct RowTableGrammarResult {
        let rows: [[Float]]
        let ids: [MLXGenerationBatchRowID]
    }

    private static func rowTableGrammarResult(
        vocabulary: [String],
        compiler: GrammarConstraintCompiler,
        configuration: GrammarSamplingConfiguration
    ) async throws -> (result: RowTableGrammarResult, events: [MLXGenerationDiagnosticEvent]) {
        var processorRows = try makeProcessorRows(
            compiler: compiler,
            configuration: configuration
        )
        let logits = MLXArray([
            Float(100), Float(110), Float(120), Float(130),
            Float(0), Float(10), Float(20), Float(30)
        ]).reshaped([2, 4])

        return try await MLXGenerationDiagnostics.withRecording {
            let masked = GrammarConstrainedLogitProcessor.processBatched(
                logits: logits,
                processorRows: &processorRows
            )
            eval(masked)
            return RowTableGrammarResult(
                rows: Self.rows(masked, columnCount: vocabulary.count),
                ids: processorRows.orderedIDs
            )
        }
    }

    private static func makeProcessorRows(
        compiler: GrammarConstraintCompiler,
        configuration: GrammarSamplingConfiguration
    ) throws -> MLXGenerationBatchRowTable<GrammarConstrainedLogitProcessor?> {
        let removedMatcher = try compiler.makeMatcher(for: configuration)
        let secondMatcher = try compiler.makeMatcher(for: configuration)
        let joinedMatcher = try compiler.makeMatcher(for: configuration)
        try secondMatcher.accept(token: 1)

        var processorRows = MLXGenerationBatchRowTable<GrammarConstrainedLogitProcessor?>()
        try processorRows.append(id: 10, payload: .init(matcher: removedMatcher))
        try processorRows.append(id: 20, payload: .init(matcher: secondMatcher))
        try processorRows.keep(ids: [20])
        try processorRows.append(id: 30, payload: .init(matcher: joinedMatcher))
        return processorRows
    }

    private static func compiler(vocabulary: [String]) throws -> GrammarConstraintCompiler {
        let directory = try makeTokenizerDirectory(vocabulary: vocabulary)
        return try GrammarConstraintCompiler(modelDirectory: directory, stopTokenIds: [0])
    }

    private static func rows(
        _ array: MLXArray,
        columnCount: Int
    ) -> [[Float]] {
        let values = array.asArray(Float.self)
        return stride(from: 0, to: values.count, by: columnCount).map { start in
            Array(values[start ..< start + columnCount])
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

    private static func makeTokenizerDirectory(vocabulary: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tokenizer = """
        {"model":{"type":"BPE","vocab":\(Self.vocabJSON(vocabulary))},"decoder":{"type":"Raw"}}
        """
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        try tokenizer.write(to: tokenizerURL, atomically: true, encoding: .utf8)
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
}
