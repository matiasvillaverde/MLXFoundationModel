@testable import MLXLocalModels
import Testing

@Suite("MLX generation batch row table")
struct MLXGenerationBatchRowTableTests {
    @Test("keeps row-local processors aligned after removal and late joins")
    func keepsRowLocalProcessorsAlignedAfterRemovalAndLateJoins() throws {
        var table = MLXGenerationBatchRowTable<RowPayload>()
        try table.append(id: 1, payload: .init(sampler: "greedy"))
        try table.append(
            id: 2,
            payload: .init(sampler: "top-p", processor: "thinking-budget")
        )
        try table.append(
            id: 3,
            payload: .init(
                sampler: "argmax",
                processor: "json-schema",
                grammarKind: .jsonSchema
            )
        )

        try table.keep(ids: [2, 3])
        try table.append(
            id: 4,
            payload: .init(
                sampler: "top-k",
                processor: "finite-choice",
                grammarKind: .choices
            )
        )

        #expect(table.orderedIDs == [2, 3, 4])
        #expect(table[2]?.processor == "thinking-budget")
        #expect(table[3]?.processor == "json-schema")
        #expect(table[3]?.grammarKind == .jsonSchema)
        #expect(table[4]?.grammarKind == .choices)
    }

    @Test("strict keep rejects duplicate and missing row ids")
    func strictKeepRejectsDuplicateAndMissingRowIDs() throws {
        var table = MLXGenerationBatchRowTable<RowPayload>()
        try table.append(id: 1, payload: .init(sampler: "greedy"))
        try table.append(id: 2, payload: .init(sampler: "top-p"))

        do {
            try table.keep(ids: [1, 1])
            Issue.record("Expected duplicate row id failure")
        } catch MLXGenerationBatchRowTableError.duplicateRowID(let id) {
            #expect(id == 1)
        }

        do {
            try table.keep(ids: [1, 3])
            Issue.record("Expected missing row id failure")
        } catch MLXGenerationBatchRowTableError.missingRowID(let id) {
            #expect(id == 3)
        }
    }

    @Test("ordered payload replacement keeps row identity")
    func orderedPayloadReplacementKeepsRowIdentity() throws {
        var table = MLXGenerationBatchRowTable<RowPayload>()
        try table.append(id: 4, payload: .init(sampler: "greedy"))
        try table.append(id: 9, payload: .init(sampler: "top-p"))

        var payloads = table.orderedPayloads
        payloads[0].processor = "json-schema"
        payloads[1].processor = "thinking-budget"
        try table.replaceOrderedPayloads(payloads)

        #expect(table.orderedIDs == [4, 9])
        #expect(table[4]?.processor == "json-schema")
        #expect(table[9]?.processor == "thinking-budget")
    }

    @Test("ordered payload replacement rejects row count drift")
    func orderedPayloadReplacementRejectsRowCountDrift() throws {
        var table = MLXGenerationBatchRowTable<RowPayload>()
        try table.append(id: 1, payload: .init(sampler: "greedy"))

        do {
            try table.replaceOrderedPayloads([])
            Issue.record("Expected row count drift to be rejected")
        } catch MLXGenerationBatchRowTableError.payloadCountMismatch(let expected, let actual) {
            #expect(expected == 1)
            #expect(actual == 0)
        }
    }

    @Test("records row operation diagnostics")
    func recordsRowOperationDiagnostics() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            var table = MLXGenerationBatchRowTable<RowPayload>()
            try table.append(id: 7, payload: .init(sampler: "greedy"))
            try table.append(id: 8, payload: .init(sampler: "top-p"))
            try table.updatePayload(for: 8) { payload in
                payload.processor = "json-schema"
                payload.grammarKind = .jsonSchema
            }
            var payloads = table.orderedPayloads
            payloads[1].processor = "finite-choice"
            try table.replaceOrderedPayloads(payloads)
            _ = table.remove(id: 7)
        }

        let snapshots: [MLXGenerationBatchRowsSnapshot] = recorded.events.compactMap { event in
            guard case .batchRows(let snapshot) = event else {
                return nil
            }
            return snapshot
        }

        #expect(snapshots.map(\.stage) == [
            .appended, .appended, .updated, .updated, .removed
        ])
        #expect(snapshots.last?.rowIDs == [8])
        #expect(snapshots.last?.affectedRowIDs == [7])
    }

    private struct RowPayload: Sendable, Equatable {
        var sampler: String
        var processor: String?
        var grammarKind: GrammarConstraintKind?

        init(
            sampler: String,
            processor: String? = nil,
            grammarKind: GrammarConstraintKind? = nil
        ) {
            self.sampler = sampler
            self.processor = processor
            self.grammarKind = grammarKind
        }
    }
}
