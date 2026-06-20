import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX oQ model artifact planner")
struct MLXOQModelArtifactPlannerTests {
    @Test("plans oQ from safetensors headers without loading tensor data")
    func plansOQFromSafetensorsHeadersWithoutLoadingTensorData() throws {
        let directory = try Self.makeModelDirectory(
            config: ["num_hidden_layers": 32],
            tensors: Dictionary(uniqueKeysWithValues: Self.defaultLayerTensors(count: 19) + [
                ("lm_head.weight", [4_096, 4_096]),
                ("model.layers.0.input_layernorm.weight", [4_096])
            ])
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let plan = try MLXOQModelArtifactPlanner.plan(
            modelDirectory: directory,
            level: "oQ4"
        )

        #expect(plan.totalParameterCount == 20 * 4_096 * 4_096)
        #expect(plan.boosts["lm_head.weight"]?.bits == 8)
        #expect(plan.decisions["model.layers.0.input_layernorm.weight"] == nil)
        #expect(plan.effectiveBitsPerWeight <= 4.7)
    }

    @Test("uses model config traits for sensitivity protection")
    func usesModelConfigTraitsForSensitivityProtection() throws {
        let projection = "model.layers.7.self_attn.q_proj.weight"
        let directory = try Self.makeModelDirectory(
            config: [
                "num_hidden_layers": 8
            ],
            tensors: [projection: [4_096, 4_096]]
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let plan = try MLXOQModelArtifactPlanner.plan(
            modelDirectory: directory,
            level: "oQ4",
            options: .init(hardCapBitsPerWeight: 6)
        )

        #expect(plan.boosts[projection]?.bits == 5)
    }

    @Test("uses nested language model traits for VLM artifacts")
    func usesNestedLanguageModelTraitsForVLMArtifacts() throws {
        let projection = "language_model.model.layers.3.self_attn.q_proj.weight"
        let directory = try Self.makeModelDirectory(
            config: [
                "architectures": ["MiniMaxM3ForConditionalGeneration"],
                "language_model": [
                    "hidden_size": 2_048,
                    "num_hidden_layers": 4,
                    "num_local_experts": 8
                ]
            ],
            tensors: [projection: [2_048, 2_048]]
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let plan = try MLXOQModelArtifactPlanner.plan(
            modelDirectory: directory,
            level: "oQ4",
            options: .init(hardCapBitsPerWeight: 6)
        )

        #expect(plan.boosts[projection]?.bits == 5)
    }

    @Test("rejects invalid oQ levels before scanning artifacts")
    func rejectsInvalidOQLevelsBeforeScanningArtifacts() throws {
        let directory = try Self.makeModelDirectory(config: [:], tensors: [:], writeWeights: false)
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try MLXOQModelArtifactPlanner.plan(modelDirectory: directory, level: "fast")
            Issue.record("Expected invalid oQ level to fail")
        } catch MLXOQModelArtifactPlannerError.invalidOQLevel(let level) {
            #expect(level == "fast")
        }
    }

    @Test("builds export manifest with MLX quantized sidecar names")
    func buildsExportManifestWithMLXQuantizedSidecarNames() throws {
        let fixture = try MLXOQExportManifestTestFixture.make()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }

        let entries = fixture.entriesBySourceName
        let quantizedEntry = try #require(entries[fixture.quantized])
        let copiedEntry = try #require(entries[fixture.copied])
        let visualEntry = try #require(entries[fixture.visual])
        let mxfp8Entry = try #require(entries[fixture.mxfp8])

        Self.assertManifestCounts(fixture.manifest)
        Self.assertAffineQuantizedEntry(quantizedEntry)
        Self.assertCopiedEntries(
            copiedEntry: copiedEntry,
            copied: fixture.copied,
            visualEntry: visualEntry,
            visual: fixture.visual
        )
        Self.assertMXFP8Entry(mxfp8Entry)
    }

    private static func assertManifestCounts(_ manifest: MLXOQExportManifest) {
        #expect(manifest.level == "oQ4")
        #expect(manifest.quantizedTensorCount == 2)
        #expect(manifest.copiedTensorCount == 2)
        #expect(manifest.sidecarTensorCount == 3)
    }

    private static func assertAffineQuantizedEntry(_ quantizedEntry: MLXOQExportTensorEntry) {
        #expect(quantizedEntry.sourceFilename == "model.safetensors")
        #expect(quantizedEntry.disposition == .quantize)
        #expect(quantizedEntry.outputNames == [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases"
        ])
    }

    private static func assertCopiedEntries(
        copiedEntry: MLXOQExportTensorEntry,
        copied: String,
        visualEntry: MLXOQExportTensorEntry,
        visual: String
    ) {
        #expect(copiedEntry.disposition == .copyFullPrecision)
        #expect(copiedEntry.outputNames == [copied])
        #expect(visualEntry.disposition == .copyFullPrecision)
        #expect(visualEntry.outputNames == [visual])
    }

    private static func assertMXFP8Entry(_ mxfp8Entry: MLXOQExportTensorEntry) {
        #expect(mxfp8Entry.quantizationSpec?.mode == "mxfp8")
        #expect(mxfp8Entry.outputNames == [
            "model.layers.1.mlp.down_proj.weight",
            "model.layers.1.mlp.down_proj.scales"
        ])
    }

    @Test("rejects directories without safetensors tensor headers")
    func rejectsDirectoriesWithoutSafetensorsTensorHeaders() throws {
        let directory = try Self.makeModelDirectory(config: [:], tensors: [:], writeWeights: false)
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try MLXOQModelArtifactPlanner.plan(modelDirectory: directory, level: "oQ4")
            Issue.record("Expected missing safetensors tensors to fail")
        } catch MLXOQModelArtifactPlannerError.noSafetensorsTensors(let url) {
            #expect(url == directory)
        }
    }

    private static func defaultLayerTensors(count: Int) -> [(String, [Int])] {
        (0..<count).map { index in
            ("model.layers.\(index).mlp.gate_proj.weight", [4_096, 4_096])
        }
    }

    private static func makeModelDirectory(
        config: [String: Any],
        tensors: [String: [Int]],
        writeWeights: Bool = true
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXOQArtifactPlanner-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try writeJSON(config, to: directory.appendingPathComponent("config.json"))
        if writeWeights {
            try writeSafetensorsHeader(
                tensors: tensors,
                to: directory.appendingPathComponent("model.safetensors")
            )
        }
        return directory
    }

    private static func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        try data.write(to: url)
    }

    private static func writeSafetensorsHeader(
        tensors: [String: [Int]],
        to url: URL
    ) throws {
        let header = tensors.reduce(into: ["__metadata__": [:]]) { result, entry in
            result[entry.key] = [
                "data_offsets": [0, 0],
                "dtype": "BF16",
                "shape": entry.value
            ]
        }
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = littleEndianUInt64Data(UInt64(headerData.count))
        data.append(headerData)
        try data.write(to: url)
    }

    private static func littleEndianUInt64Data(_ value: UInt64) -> Data {
        var littleEndian = value.littleEndian
        return Data(bytes: &littleEndian, count: MemoryLayout<UInt64>.size)
    }
}
