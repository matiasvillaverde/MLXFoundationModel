import Foundation
import MLX
@testable import MLXFoundationModel
import Testing

@Suite("MLX oQ model artifact converter")
struct MLXOQModelArtifactConverterTests {
    @Test("writes MLX-compatible quantized safetensors and config")
    func writesMLXCompatibleQuantizedSafetensorsAndConfig() throws {
        try Device.withDefaultDevice(.cpu) {
            let fixture = try MLXOQModelArtifactConverterTestFixture.make()
            defer { try? FileManager.default.removeItem(at: fixture.rootDirectory) }

            let manifest = try MLXOQModelArtifactConverter.convert(
                modelDirectory: fixture.modelDirectory,
                outputDirectory: fixture.outputDirectory,
                level: "oQ4"
            )
            let arrays = try Self.convertedArrays(in: fixture)
            let config = try Self.convertedConfig(in: fixture)

            Self.assertManifest(manifest)
            Self.assertConvertedArrays(arrays)
            Self.assertConvertedConfig(config)
            Self.assertTokenizerCopied(in: fixture)
        }
    }

    @Test("refuses an existing output directory by default")
    func refusesExistingOutputDirectoryByDefault() throws {
        let fixture = try MLXOQModelArtifactConverterTestFixture.make()
        defer { try? FileManager.default.removeItem(at: fixture.rootDirectory) }
        try FileManager.default.createDirectory(
            at: fixture.outputDirectory,
            withIntermediateDirectories: true
        )

        do {
            _ = try MLXOQModelArtifactConverter.convert(
                modelDirectory: fixture.modelDirectory,
                outputDirectory: fixture.outputDirectory,
                level: "oQ4"
            )
            Issue.record("Expected existing output directory to fail")
        } catch MLXOQModelArtifactConverterError.outputDirectoryExists(let url) {
            #expect(url == fixture.outputDirectory)
        }
    }

    private static func assertManifest(_ manifest: MLXOQExportManifest) {
        #expect(manifest.level == "oQ4")
        #expect(manifest.quantizedTensorCount == 1)
        #expect(manifest.copiedTensorCount == 2)
    }

    private static func assertConvertedArrays(_ arrays: [String: MLXArray]) {
        let tensor = MLXOQModelArtifactConverterTestFixture.self
        #expect(arrays.keys.contains(tensor.quantizedTensorName))
        #expect(arrays.keys.contains("model.layers.0.self_attn.q_proj.scales"))
        #expect(arrays.keys.contains("model.layers.0.self_attn.q_proj.biases"))
        #expect(arrays.keys.contains(tensor.copiedTensorName))
        #expect(arrays.keys.contains(tensor.visualTensorName))
    }

    private static func assertConvertedConfig(_ config: [String: Any]) {
        let quantization = config["quantization"] as? [String: Any]
        let layer = quantization?["model.layers.0.self_attn.q_proj"] as? [String: Any]
        let metadata = config["mlx_oq"] as? [String: Any]
        #expect(quantization?["group_size"] as? Int == 64)
        #expect(layer?["bits"] as? Int == 4)
        #expect(layer?["quantization_mode"] as? String == "affine")
        #expect(metadata?["level"] as? String == "oQ4")
    }

    private static func assertTokenizerCopied(in fixture: MLXOQModelArtifactConverterTestFixture) {
        let tokenizerURL = fixture.outputDirectory
            .appendingPathComponent(MLXOQModelArtifactConverterTestFixture.tokenizerFilename)
        #expect(FileManager.default.fileExists(atPath: tokenizerURL.path))
    }

    private static func convertedArrays(
        in fixture: MLXOQModelArtifactConverterTestFixture
    ) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: fixture.outputDirectory.appendingPathComponent("model.safetensors"))
    }

    private static func convertedConfig(
        in fixture: MLXOQModelArtifactConverterTestFixture
    ) throws -> [String: Any] {
        let url = fixture.outputDirectory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
