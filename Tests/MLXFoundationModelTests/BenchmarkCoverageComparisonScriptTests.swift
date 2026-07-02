import Foundation
import Testing

@Suite("Benchmark coverage comparison script")
struct BenchmarkCoverageComparisonScriptTests {
    @Test("passes when benchmark coverage is preserved")
    func passesForStableBenchmarkCoverage() throws {
        let fixture = try Self.makeFixture(currentStatus: "passed")
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try Self.runCompare(fixture: fixture)
        let report = try Self.report(from: result.output)
        let rows = try #require(report["benchmark_coverage_rows"] as? [[String: Any]])
        let row = try #require(rows.first)

        #expect(result.status == 0, Comment(rawValue: result.output))
        #expect(report["passed"] as? Bool == true)
        #expect(row["status"] as? String == "passed")
    }

    @Test("fails when benchmark coverage regresses")
    func failsForBenchmarkCoverageRegression() throws {
        let fixture = try Self.makeFixture(currentStatus: "missing")
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try Self.runCompare(fixture: fixture)
        let report = try Self.report(from: result.output)
        let coverageRows = try #require(
            report["benchmark_coverage_rows"] as? [[String: Any]]
        )
        let summaryRows = try #require(
            report["coverage_summary_rows"] as? [[String: Any]]
        )
        let coverageRow = try #require(coverageRows.first)
        let summaryRow = try #require(summaryRows.first { row in
            row["key"] as? String == "benchmark_coverage"
        })

        #expect(result.status == 1, Comment(rawValue: result.output))
        #expect(report["passed"] as? Bool == false)
        #expect(coverageRow["status"] as? String == "benchmark_coverage_regressed")
        #expect(summaryRow["status"] as? String == "coverage_summary_regressed")
    }

    private struct Fixture {
        let directory: URL
        let baseline: URL
        let current: URL
    }

    private struct ProcessResult {
        let status: Int32
        let output: String
    }

    private static func makeFixture(currentStatus: String) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("BenchmarkCoverageComparison-\(UUID().uuidString)")
        let baseline = directory.appendingPathComponent("baseline.json")
        let current = directory.appendingPathComponent("current.json")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.writeSummary(benchmarkStatus: "passed", to: baseline)
        try Self.writeSummary(benchmarkStatus: currentStatus, to: current)
        return Fixture(directory: directory, baseline: baseline, current: current)
    }

    private static func writeSummary(benchmarkStatus: String, to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: Self.summary(benchmarkStatus: benchmarkStatus),
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: url)
    }

    private static func summary(benchmarkStatus: String) -> [String: Any] {
        [
            "benchmark_parse_errors": [],
            "benchmark_records": [Self.benchmarkRecord()],
            "tests": [Self.testRecord()],
            "feature_coverage": Self.featureCoverage(),
            "benchmark_coverage": Self.benchmarkCoverage(status: benchmarkStatus)
        ]
    }

    private static func benchmarkRecord() -> [String: Any] {
        [
            "kind": "bench",
            "model": "qwen",
            "architecture": "qwen3",
            "decode_tps": 10.0,
            "total_tps": 20.0,
            "prompt_tps": 30.0,
            "e2e_tps": 40.0
        ]
    }

    private static func testRecord() -> [String: Any] {
        [
            "label": "qwen generation",
            "status": "passed",
            "duration_seconds": 10.0
        ]
    }

    private static func featureCoverage() -> [String: Any] {
        [
            "passed": true,
            "status": "available",
            "selected_model_metadata_count": 1,
            "rows": [
                [
                    "model_id": "qwen",
                    "architecture": "qwen3",
                    "feature_key": "generation",
                    "status": "passed"
                ]
            ]
        ]
    }

    private static func benchmarkCoverage(status: String) -> [String: Any] {
        [
            "passed": status == "passed",
            "status": status == "passed" ? "available" : "selected_model_count_mismatch",
            "selected_model_metadata_count": 1,
            "rows": [
                [
                    "model_id": "qwen",
                    "architecture": "qwen3",
                    "status": status,
                    "benchmark_count": status == "passed" ? 1 : 0
                ]
            ]
        ]
    }

    private static func runCompare(fixture: Fixture) throws -> ProcessResult {
        let output = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            Self.scriptURL.path,
            "--baseline",
            fixture.baseline.path,
            "--current",
            fixture.current.path,
            "--json"
        ]
        process.standardOutput = output
        process.standardError = output
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return ProcessResult(
            status: process.terminationStatus,
            output: String(data: data, encoding: .utf8) ?? ""
        )
    }

    private static func report(from output: String) throws -> [String: Any] {
        let data = try #require(output.data(using: .utf8))
        let object = try JSONSerialization.jsonObject(with: data)
        return try #require(object as? [String: Any])
    }

    private static var scriptURL: URL {
        repositoryRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("compare-benchmark-summaries.py")
    }

    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
