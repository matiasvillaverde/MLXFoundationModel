import Foundation
import Testing

enum RealModelSummaryFixture {
    struct Fixture {
        let directory: URL
        let log: URL
        let summary: URL
    }

    static func makeFixture() throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RealModelSummaryScriptTests-\(UUID().uuidString)")
        let log = directory.appendingPathComponent("real-model.log")
        let summary = directory.appendingPathComponent("summary.json")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return Fixture(directory: directory, log: log, summary: summary)
    }

    static func completeLogWithBenchmark() throws -> String {
        var lines = ["MLXFoundationModel real-model validation"]
        for featureKey in Self.attentionFeatureKeys() {
            lines.append(contentsOf: try Self.testLines(featureKey: featureKey))
        }
        lines.append(try Self.benchmarkLine())
        return lines.joined(separator: "\n") + "\n"
    }

    static func completeLogWithoutBenchmark() throws -> String {
        var lines = ["MLXFoundationModel real-model validation"]
        for featureKey in Self.attentionFeatureKeys() {
            lines.append(contentsOf: try Self.testLines(featureKey: featureKey))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    static func multiFeatureLogWithBenchmark() throws -> String {
        var lines = ["MLXFoundationModel real-model validation"]
        for featureKey in Self.attentionFeatureKeys() {
            guard !Self.nativeToolFeatureKeys.contains(featureKey) else {
                continue
            }
            lines.append(contentsOf: try Self.testLines(featureKey: featureKey))
        }
        lines.append(contentsOf: try Self.testLines(
            label: "demo native tool combined",
            featureKeys: Self.nativeToolFeatureKeys
        ))
        lines.append(try Self.benchmarkLine())
        return lines.joined(separator: "\n") + "\n"
    }

    static func incompleteLogWithBenchmark() throws -> String {
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
            "stop_sequence",
            "continuous_batch_prompt_cache",
            "persistent_prompt_cache",
            "json_schema_constraints",
            "finite_choice_constraints",
            "token_grammar_constraints",
            "tool_call_rendering",
            "tool_stream_translation",
            "tool_schema_normalization",
            "native_tool_constraints",
            "native_tool_stream_translation",
            "runtime_kv_cache",
            "session_style_request",
            "rendered_text_streaming"
        ]
    }

    private static let nativeToolFeatureKeys = [
        "tool_call_rendering",
        "tool_stream_translation",
        "tool_schema_normalization",
        "native_tool_constraints",
        "native_tool_stream_translation"
    ]

    private static func testLines(featureKey: String) throws -> [String] {
        try Self.testLines(
            label: "demo \(featureKey)",
            featureKeys: [featureKey]
        )
    }

    private static func testLines(label: String, featureKeys: [String]) throws -> [String] {
        let metadata = try Self.metadata(label: label, featureKeys: featureKeys)
        return [
            "-> \(label)",
            "TEST_META_JSON \(metadata)",
            "   passed",
            "   duration_seconds: 1"
        ]
    }

    private static func metadata(label: String, featureKeys: [String]) throws -> String {
        let object: [String: Any] = [
            "schema_version": 1,
            "label": label,
            "model_id": "demo-model",
            "feature_key": featureKeys.first ?? "",
            "feature_keys": featureKeys,
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

    static func writeLog(_ value: String, to url: URL) throws {
        try value.write(to: url, atomically: true, encoding: .utf8)
    }

    static func runSummary(
        fixture: Fixture,
        selectedCount: String
    ) throws {
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
            selectedCount,
            "--swift-test-invocation-count",
            "14"
        ]
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
    }

    static func requiredMismatchRow(in rows: [[String: Any]]) throws -> [String: Any] {
        try #require(rows.first { row in
            row["status"] as? String == "selected_model_count_mismatch"
        })
    }

    static func featureRow(
        _ featureKey: String,
        in rows: [[String: Any]]
    ) throws -> [String: Any] {
        try #require(rows.first { row in
            row["feature_key"] as? String == featureKey
        })
    }

    static func coverage(from url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data)
        let summary = try #require(object as? [String: Any])
        return try #require(summary["feature_coverage"] as? [String: Any])
    }

    static func benchmarkCoverage(from url: URL) throws -> [String: Any] {
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
