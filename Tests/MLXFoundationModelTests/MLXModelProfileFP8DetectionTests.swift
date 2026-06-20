import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model profile FP8 detection")
struct MLXModelProfileFP8DetectionTests {
    @Test("loads FP8 scale sidecar evidence from safetensors index")
    func loadsFP8ScaleSidecarEvidenceFromSafetensorsIndex() throws {
        let directory = try Self.makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeJSON(
            Self.fp8ScaleIndex,
            to: directory.appendingPathComponent("model.safetensors.index.json")
        )

        let optimization = try Self.optimizationProfile(
            id: "glm4-moe-fp8-sidecar-fixture",
            location: directory
        )

        Self.expectFP8ScaleDequantization(optimization)
    }

    @Test("loads FP8 scale sidecar evidence from safetensors header")
    func loadsFP8ScaleSidecarEvidenceFromSafetensorsHeader() throws {
        let directory = try Self.makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeSafetensorsHeader(
            tensors: [
                .init(
                    key: "model.layers.0.mlp.experts.0.gate_proj.weight",
                    dtype: "F16",
                    shape: [64, 128]
                ),
                .init(
                    key: "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
                    dtype: "F16",
                    shape: [2, 4]
                )
            ],
            to: directory.appendingPathComponent("model.safetensors")
        )

        let optimization = try Self.optimizationProfile(
            id: "glm4-moe-fp8-header-fixture",
            location: directory
        )

        Self.expectFP8ScaleDequantization(optimization)
    }

    @Test("loads FP8 dot-scale pair evidence from safetensors header")
    func loadsFP8DotScalePairEvidenceFromSafetensorsHeader() throws {
        let directory = try Self.makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeSafetensorsHeader(
            tensors: [
                .init(
                    key: "model.layers.0.mlp.experts.0.down_proj.weight",
                    dtype: "F8_E4M3",
                    shape: [64, 128]
                ),
                .init(
                    key: "model.layers.0.mlp.experts.0.down_proj.scale",
                    dtype: "F8_E8M0",
                    shape: [2, 4]
                )
            ],
            to: directory.appendingPathComponent("model.safetensors")
        )

        let optimization = try Self.optimizationProfile(
            id: "glm4-moe-fp8-dot-scale-fixture",
            location: directory
        )

        Self.expectFP8ScaleDequantization(optimization)
    }

    @Test("ignores non-FP8 dot-scale header pairs")
    func ignoresNonFP8DotScaleHeaderPairs() throws {
        let directory = try Self.makeTemporaryModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeSafetensorsHeader(
            tensors: [
                .init(
                    key: "model.layers.0.mlp.shared_expert.down_proj.weight",
                    dtype: "F16",
                    shape: [64, 128]
                ),
                .init(
                    key: "model.layers.0.mlp.shared_expert.down_proj.scale",
                    dtype: "F16",
                    shape: [2, 4]
                )
            ],
            to: directory.appendingPathComponent("model.safetensors")
        )

        let optimization = try Self.optimizationProfile(
            id: "glm4-moe-f16-dot-scale-fixture",
            location: directory
        )

        #expect(!optimization.requiresFP8ScaleDequantization)
        #expect(!optimization.detectedFeatures.contains(.fp8ScaleDequantization))
        #expect(!optimization.implementedFeatures.contains(.fp8ScaleDequantization))
    }

    private struct SafetensorsTensorHeader {
        let key: String
        let dtype: String
        let shape: [Int]
    }

    private static var fp8ScaleIndex: [String: Any] {
        [
            "metadata": [:],
            "weight_map": [
                "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00001.safetensors",
                "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": "model-00001.safetensors"
            ]
        ]
    }

    private static var glmMoEConfig: [String: Any] {
        [
            "model_type": "glm4_moe",
            "architectures": ["Glm4MoeForCausalLM"],
            "num_experts": 32
        ]
    }

    private static func makeTemporaryModelDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "MLXModelProfileFP8DetectionTests-\(UUID().uuidString)",
                isDirectory: true
            )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try writeJSON(Self.glmMoEConfig, to: directory.appendingPathComponent("config.json"))
        try writeJSON([:], to: directory.appendingPathComponent("tokenizer_config.json"))
        return directory
    }

    private static func optimizationProfile(
        id: String,
        location: URL
    ) throws -> MLXModelOptimizationProfile {
        let profile = try #require(try MLXModel.profiled(id: id, location: location).profile)
        return profile.optimizationProfile
    }

    private static func expectFP8ScaleDequantization(
        _ optimization: MLXModelOptimizationProfile
    ) {
        #expect(optimization.requiresFP8ScaleDequantization)
        #expect(optimization.detectedFeatures.contains(.fp8ScaleDequantization))
        #expect(optimization.implementedFeatures.contains(.fp8ScaleDequantization))
    }

    private static func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        try data.write(to: url)
    }

    private static func writeSafetensorsHeader(
        tensors: [SafetensorsTensorHeader],
        to url: URL
    ) throws {
        var header: [String: Any] = ["__metadata__": [:]]
        for (index, tensor) in tensors.enumerated() {
            header[tensor.key] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [index * 2, (index + 1) * 2]
            ]
        }
        let headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys]
        )
        var data = littleEndianUInt64Data(UInt64(headerData.count))
        data.append(headerData)
        data.append(Data(repeating: 0, count: tensors.count * 2))
        try data.write(to: url)
    }

    private static func littleEndianUInt64Data(_ value: UInt64) -> Data {
        var bytes = [UInt8]()
        bytes.reserveCapacity(8)
        for offset in 0 ..< 8 {
            bytes.append(UInt8((value >> UInt64(offset * 8)) & 0xff))
        }
        return Data(bytes)
    }
}
