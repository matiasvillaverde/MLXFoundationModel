import Foundation
import Testing

@Suite("Real-model summary script")
struct RealModelSummaryScriptTests {
    @Test("marks coverage passed when every required feature passed")
    func passesForCompleteCoverage() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        try Self.writeLog(try Self.completeLogWithBenchmark(), to: fixture.log)
        try Self.runSummary(fixture: fixture)
        let coverage = try Self.coverage(from: fixture.summary)
        let benchmarkCoverage = try Self.benchmarkCoverage(from: fixture.summary)
        let rows = try #require(coverage["rows"] as? [[String: Any]])

        #expect(coverage["passed"] as? Bool == true)
        #expect(benchmarkCoverage["passed"] as? Bool == true)
        #expect(rows.count == 14)
    }

    @Test("marks coverage failed when a required feature is missing")
    func failsForMissingRequiredFeature() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        try Self.writeLog(try Self.incompleteLogWithBenchmark(), to: fixture.log)
        try Self.runSummary(fixture: fixture)
        let coverage = try Self.coverage(from: fixture.summary)
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
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        try Self.writeLog(try Self.completeLogWithoutBenchmark(), to: fixture.log)
        try Self.runSummary(fixture: fixture)
        let coverage = try Self.coverage(from: fixture.summary)
        let benchmarkCoverage = try Self.benchmarkCoverage(from: fixture.summary)
        let rows = try #require(benchmarkCoverage["rows"] as? [[String: Any]])
        let row = try #require(rows.first)

        #expect(coverage["passed"] as? Bool == true)
        #expect(benchmarkCoverage["passed"] as? Bool == false)
        #expect(row["status"] as? String == "missing")
    }

    private struct Fixture {
        let directory: URL
        let log: URL
        let summary: URL
    }

    private static func makeFixture() throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RealModelSummaryScriptTests-\(UUID().uuidString)")
        let log = directory.appendingPathComponent("real-model.log")
        let summary = directory.appendingPathComponent("summary.json")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return Fixture(directory: directory, log: log, summary: summary)
    }

    private static func completeLogWithBenchmark() throws -> String {
        var lines = ["MLXFoundationModel real-model validation"]
        for featureKey in Self.attentionFeatureKeys() {
            lines.append(contentsOf: try Self.testLines(featureKey: featureKey))
        }
        lines.append(try Self.benchmarkLine())
        return lines.joined(separator: "\n") + "\n"
    }

    private static func completeLogWithoutBenchmark() throws -> String {
        var lines = ["MLXFoundationModel real-model validation"]
        for featureKey in Self.attentionFeatureKeys() {
            lines.append(contentsOf: try Self.testLines(featureKey: featureKey))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func incompleteLogWithBenchmark() throws -> String {
        var lines = ["MLXFoundationModel real-model validation"]
        for featureKey in Self.attentionFeatureKeys() where featureKey != "sampling_logits" {
            lines.append(contentsOf: try Self.testLines(featureKey: featureKey))
        }
        lines.append(try Self.benchmarkLine())
        return lines.joined(separator: "\n") + "\n"
    }

    private static func attentionFeatureKeys() -> [String] {
        [
            "generation",
            "sampling_logits",
            "stream_lifecycle",
            "model_load_progress",
            "memory_guard",
            "observability",
            "greedy_constrained_decode",
            "continuous_batch_prompt_cache",
            "persistent_prompt_cache",
            "json_schema_constraints",
            "finite_choice_constraints",
            "token_grammar_constraints",
            "runtime_kv_cache",
            "session_style_request"
        ]
    }

    private static func testLines(featureKey: String) throws -> [String] {
        let label = "demo \(featureKey)"
        let metadata = try Self.metadata(label: label, featureKey: featureKey)
        return [
            "-> \(label)",
            "TEST_META_JSON \(metadata)",
            "   passed",
            "   duration_seconds: 1"
        ]
    }

    private static func metadata(label: String, featureKey: String) throws -> String {
        let object: [String: Any] = [
            "schema_version": 1,
            "label": label,
            "model_id": "demo-model",
            "feature_key": featureKey,
            "architecture": "qwen3",
            "tags": []
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return try #require(String(data: data, encoding: .utf8))
    }

    private static func benchmarkLine() throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: Self.benchmarkRecord(),
            options: [.sortedKeys]
        )
        let json = try #require(String(data: data, encoding: .utf8))
        return "BENCH_JSON \(json)"
    }

    private static func benchmarkRecord() -> [String: Any] {
        [
            "schema_version": 1,
            "kind": "bench",
            "model": "demo-model",
            "architecture": "qwen3",
            "decode_tps": 10.0,
            "prompt_tps": 20.0,
            "total_tps": 30.0,
            "e2e_tps": 40.0
        ]
    }

    private static func writeLog(_ value: String, to url: URL) throws {
        try value.write(to: url, atomically: true, encoding: .utf8)
    }

    private static func runSummary(fixture: Fixture) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            Self.scriptURL.path,
            "--log",
            fixture.log.path,
            "--summary",
            fixture.summary.path,
            "--result",
            "passed",
            "--started-utc",
            "2026-07-02T00:00:00Z",
            "--catalog",
            "catalog.json",
            "--models",
            ".models",
            "--scope",
            "test",
            "--host-memory-gb",
            "32",
            "--selected-count",
            "1",
            "--swift-test-invocation-count",
            "14"
        ]
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
    }

    private static func coverage(from url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data)
        let summary = try #require(object as? [String: Any])
        return try #require(summary["feature_coverage"] as? [String: Any])
    }

    private static func benchmarkCoverage(from url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data)
        let summary = try #require(object as? [String: Any])
        return try #require(summary["benchmark_coverage"] as? [String: Any])
    }

    private static var scriptURL: URL {
        repositoryRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("summarize-real-model-run.py")
    }

    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
