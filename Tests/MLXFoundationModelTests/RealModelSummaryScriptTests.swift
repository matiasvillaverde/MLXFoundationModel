import Foundation
import Testing

@Suite("Real-model summary script")
struct RealModelSummaryScriptTests {
    @Test("marks coverage passed when every required feature passed")
    func passesForCompleteCoverage() throws {
        let fixture = try RealModelSummaryFixture.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let log = try RealModelSummaryFixture.completeLogWithBenchmark()
        try RealModelSummaryFixture.writeLog(log, to: fixture.log)
        try RealModelSummaryFixture.runSummary(fixture: fixture, selectedCount: "1")
        let coverage = try RealModelSummaryFixture.coverage(from: fixture.summary)
        let benchmarkCoverage = try RealModelSummaryFixture.benchmarkCoverage(from: fixture.summary)
        let rows = try #require(coverage["rows"] as? [[String: Any]])

        #expect(coverage["passed"] as? Bool == true)
        #expect(benchmarkCoverage["passed"] as? Bool == true)
        #expect(rows.count == 18)
    }

    @Test("marks coverage failed when a required feature is missing")
    func failsForMissingRequiredFeature() throws {
        let fixture = try RealModelSummaryFixture.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let log = try RealModelSummaryFixture.incompleteLogWithBenchmark()
        try RealModelSummaryFixture.writeLog(log, to: fixture.log)
        try RealModelSummaryFixture.runSummary(fixture: fixture, selectedCount: "1")
        let coverage = try RealModelSummaryFixture.coverage(from: fixture.summary)
        let rows = try #require(coverage["rows"] as? [[String: Any]])
        let missing = rows.filter { row in
            row["status"] as? String == "missing"
        }

        #expect(coverage["passed"] as? Bool == false)
        #expect(missing.contains { row in
            row["feature_key"] as? String == "sampling_logits"
        })
    }

    @Test("marks benchmark coverage failed when generation emits no benchmark record")
    func failsForMissingBenchmarkRecord() throws {
        let fixture = try RealModelSummaryFixture.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let log = try RealModelSummaryFixture.completeLogWithoutBenchmark()
        try RealModelSummaryFixture.writeLog(log, to: fixture.log)
        try RealModelSummaryFixture.runSummary(fixture: fixture, selectedCount: "1")
        let coverage = try RealModelSummaryFixture.coverage(from: fixture.summary)
        let benchmarkCoverage = try RealModelSummaryFixture.benchmarkCoverage(from: fixture.summary)
        let rows = try #require(benchmarkCoverage["rows"] as? [[String: Any]])
        let row = try #require(rows.first)

        #expect(coverage["passed"] as? Bool == true)
        #expect(benchmarkCoverage["passed"] as? Bool == false)
        #expect(row["status"] as? String == "missing")
    }

    @Test("marks multiple features passed from one test metadata record")
    func passesMultipleFeaturesFromOneTestMetadataRecord() throws {
        let fixture = try RealModelSummaryFixture.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let log = try RealModelSummaryFixture.multiFeatureLogWithBenchmark()
        try RealModelSummaryFixture.writeLog(log, to: fixture.log)
        try RealModelSummaryFixture.runSummary(fixture: fixture, selectedCount: "1")
        let coverage = try RealModelSummaryFixture.coverage(from: fixture.summary)
        let rows = try #require(coverage["rows"] as? [[String: Any]])
        let nativeConstraints = try RealModelSummaryFixture.featureRow("native_tool_constraints", in: rows)
        let nativeStreaming = try RealModelSummaryFixture.featureRow(
            "native_tool_stream_translation",
            in: rows
        )

        #expect(coverage["passed"] as? Bool == true)
        #expect(nativeConstraints["status"] as? String == "passed")
        #expect(nativeStreaming["status"] as? String == "passed")
        #expect(nativeConstraints["label"] as? String == "demo native tool combined")
        #expect(nativeStreaming["label"] as? String == "demo native tool combined")
    }

    @Test("marks coverage failed when selected model metadata is incomplete")
    func failsForSelectedModelMetadataCountMismatch() throws {
        let fixture = try RealModelSummaryFixture.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let log = try RealModelSummaryFixture.completeLogWithBenchmark()
        try RealModelSummaryFixture.writeLog(log, to: fixture.log)
        try RealModelSummaryFixture.runSummary(fixture: fixture, selectedCount: "2")
        let coverage = try RealModelSummaryFixture.coverage(from: fixture.summary)
        let benchmarkCoverage = try RealModelSummaryFixture.benchmarkCoverage(from: fixture.summary)
        let featureRows = try #require(coverage["rows"] as? [[String: Any]])
        let benchmarkRows = try #require(benchmarkCoverage["rows"] as? [[String: Any]])
        let featureMismatch = try RealModelSummaryFixture.requiredMismatchRow(in: featureRows)
        let benchmarkMismatch = try RealModelSummaryFixture.requiredMismatchRow(in: benchmarkRows)

        #expect(coverage["passed"] as? Bool == false)
        #expect(benchmarkCoverage["passed"] as? Bool == false)
        #expect(featureMismatch["selected_model_count"] as? Int == 2)
        #expect(benchmarkMismatch["selected_model_metadata_count"] as? Int == 1)
    }
}
