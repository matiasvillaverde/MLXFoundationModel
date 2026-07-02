import Foundation
import Testing

@Suite("Benchmark comparison script")
struct BenchmarkComparisonScriptTests {
    @Test("passes when feature durations stay within threshold")
    func passesForStableFeatureDurations() throws {
        let fixture = try Self.makeFixture(baselineDuration: 10, currentDuration: 12)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try Self.runCompare(fixture: fixture, maxRatio: "1.50")
        let report = try Self.report(from: result.output)
        let rows = try #require(report["test_rows"] as? [[String: Any]])
        let row = try #require(rows.first)

        #expect(result.status == 0, Comment(rawValue: result.output))
        #expect(report["passed"] as? Bool == true)
        #expect(row["status"] as? String == "passed")
    }

    @Test("fails when feature duration exceeds threshold")
    func failsForDurationRegression() throws {
        let fixture = try Self.makeFixture(baselineDuration: 10, currentDuration: 20)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try Self.runCompare(fixture: fixture, maxRatio: "1.50")
        let report = try Self.report(from: result.output)
        let rows = try #require(report["test_rows"] as? [[String: Any]])
        let row = try #require(rows.first)

        #expect(result.status == 1, Comment(rawValue: result.output))
        #expect(report["passed"] as? Bool == false)
        #expect(row["status"] as? String == "duration_regressed")
    }

    @Test("passes when feature coverage is preserved")
    func passesForStableFeatureCoverage() throws {
        let fixture = try Self.makeCoverageFixture(currentCoverageStatus: "passed")
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try Self.runCompare(fixture: fixture, maxRatio: "1.50")
        let report = try Self.report(from: result.output)
        let rows = try #require(report["coverage_rows"] as? [[String: Any]])
        let row = try #require(rows.first)

        #expect(result.status == 0, Comment(rawValue: result.output))
        #expect(report["passed"] as? Bool == true)
        #expect(row["status"] as? String == "passed")
    }

    @Test("fails when feature coverage regresses")
    func failsForFeatureCoverageRegression() throws {
        let fixture = try Self.makeCoverageFixture(currentCoverageStatus: "missing")
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let result = try Self.runCompare(fixture: fixture, maxRatio: "1.50")
        let report = try Self.report(from: result.output)
        let rows = try #require(report["coverage_rows"] as? [[String: Any]])
        let row = try #require(rows.first)

        #expect(result.status == 1, Comment(rawValue: result.output))
        #expect(report["passed"] as? Bool == false)
        #expect(row["status"] as? String == "coverage_regressed")
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

    private static func makeFixture(
        baselineDuration: Double,
        currentDuration: Double
    ) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("BenchmarkComparisonScriptTests-\(UUID().uuidString)")
        let baseline = directory.appendingPathComponent("baseline.json")
        let current = directory.appendingPathComponent("current.json")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.writeSummary(duration: baselineDuration, to: baseline)
        try Self.writeSummary(duration: currentDuration, to: current)
        return Fixture(directory: directory, baseline: baseline, current: current)
    }

    private static func makeCoverageFixture(currentCoverageStatus: String) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("BenchmarkComparisonScriptTests-\(UUID().uuidString)")
        let baseline = directory.appendingPathComponent("baseline.json")
        let current = directory.appendingPathComponent("current.json")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.writeSummary(duration: 10, coverageStatus: "passed", to: baseline)
        try Self.writeSummary(duration: 10, coverageStatus: currentCoverageStatus, to: current)
        return Fixture(directory: directory, baseline: baseline, current: current)
    }

    private static func writeSummary(duration: Double, to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: Self.summary(duration: duration),
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: url)
    }

    private static func writeSummary(
        duration: Double,
        coverageStatus: String,
        to url: URL
    ) throws {
        let data = try JSONSerialization.data(
            withJSONObject: Self.summary(duration: duration, coverageStatus: coverageStatus),
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: url)
    }

    private static func summary(duration: Double) -> [String: Any] {
        [
            "benchmark_parse_errors": [],
            "benchmark_records": [Self.benchmarkRecord()],
            "tests": [
                [
                    "label": "qwen feature",
                    "status": "passed",
                    "duration_seconds": duration
                ]
            ]
        ]
    }

    private static func summary(duration: Double, coverageStatus: String) -> [String: Any] {
        var value = Self.summary(duration: duration)
        value["feature_coverage"] = Self.featureCoverage(status: coverageStatus)
        return value
    }

    private static func featureCoverage(status: String) -> [String: Any] {
        [
            "schema_version": 1,
            "passed": status == "passed",
            "status": "available",
            "rows": [
                [
                    "model_id": "qwen",
                    "architecture": "qwen3",
                    "feature_key": "generation",
                    "status": status,
                    "test_status": status,
                    "label": "qwen generation",
                    "duration_seconds": 10.0
                ]
            ]
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

    private static func runCompare(
        fixture: Fixture,
        maxRatio: String
    ) throws -> ProcessResult {
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
            "--test-duration-max-ratio",
            maxRatio,
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
