import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model profile MTP detection")
struct MLXModelProfileMTPDetectionTests {
    @Test("does not infer native MTP from family name without MTP heads")
    func doesNotInferNativeMTPFromFamilyNameWithoutMTPHeads() {
        let optimization = MLXModelProfile.make(
            config: [
                "model_type": "deepseek_v4",
                "architectures": ["DeepseekV4ForCausalLM"]
            ],
            id: "DeepSeek-V4-Flash"
        )
        .optimizationProfile

        #expect(!optimization.hasNativeMTPWeights)
        #expect(!optimization.supportsNativeMTP)
        #expect(!optimization.detectedFeatures.contains(.nativeMTP))
    }

    @Test("does not infer native MTP from config declaration without tensors")
    func doesNotInferNativeMTPFromConfigDeclarationWithoutTensors() {
        let optimization = MLXModelProfile.make(
            config: [
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": [
                    "model_type": "qwen3_5_text",
                    "mtp_num_hidden_layers": 1
                ]
            ],
            id: "qwen3.5-config-only"
        )
        .optimizationProfile

        #expect(!optimization.hasNativeMTPWeights)
        #expect(!optimization.supportsNativeMTP)
        #expect(!optimization.nativeMTPRuntimeSupported)
        #expect(!optimization.detectedFeatures.contains(.nativeMTP))
        #expect(!optimization.implementedFeatures.contains(.nativeMTP))
    }

    @Test("loads native MTP tensor evidence from safetensors index")
    func loadsNativeMTPTensorEvidenceFromSafetensorsIndex() throws {
        let directory = try Self.makeTemporaryModelDirectory(
            config: Self.qwenConfig(modelType: "qwen3_5"),
            tokenizerConfig: [:]
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeJSON(
            Self.mtpIndex,
            to: directory.appendingPathComponent("model.safetensors.index.json")
        )

        let optimization = try Self.optimizationProfile(
            id: "qwen3.5-mtp-fixture",
            location: directory
        )

        Self.expectNativeMTPSupported(optimization, runtimeSupported: true)
    }

    @Test("loads native MTP tensor evidence from safetensors header")
    func loadsNativeMTPTensorEvidenceFromSafetensorsHeader() throws {
        let directory = try Self.makeTemporaryModelDirectory(
            config: Self.qwenConfig(modelType: "qwen3_6"),
            tokenizerConfig: [:]
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeSafetensorsHeader(
            keys: [
                "model.mtp.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.weight"
            ],
            to: directory.appendingPathComponent("model.safetensors")
        )

        let optimization = try Self.optimizationProfile(
            id: "qwen3.6-mtp-header-fixture",
            location: directory
        )

        Self.expectNativeMTPSupported(optimization, runtimeSupported: false)
    }

    @Test("decodes legacy profile JSON without native MTP runtime support flag")
    func decodesLegacyProfileJSONWithoutNativeMTPRuntimeSupportFlag() throws {
        let profile = try JSONDecoder().decode(
            MLXModelOptimizationProfile.self,
            from: Data("""
            {
                "hasNativeMTPWeights": true,
                "supportsNativeMTP": true
            }
            """.utf8)
        )

        #expect(profile.hasNativeMTPWeights)
        #expect(profile.supportsNativeMTP)
        #expect(!profile.nativeMTPRuntimeSupported)
        #expect(profile.detectedFeatures.contains(.nativeMTP))
        #expect(!profile.implementedFeatures.contains(.nativeMTP))
    }

    private static var mtpIndex: [String: Any] {
        [
            "metadata": [:],
            "weight_map": [
                "model.mtp.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
            ]
        ]
    }

    private static func qwenConfig(modelType: String) -> [String: Any] {
        [
            "model_type": modelType,
            "architectures": ["Qwen3ForCausalLM"]
        ]
    }

    private static func optimizationProfile(
        id: String,
        location: URL
    ) throws -> MLXModelOptimizationProfile {
        let profile = try #require(try MLXModel.profiled(id: id, location: location).profile)
        return profile.optimizationProfile
    }

    private static func expectNativeMTPSupported(
        _ optimization: MLXModelOptimizationProfile,
        runtimeSupported: Bool
    ) {
        #expect(optimization.hasNativeMTPWeights)
        #expect(optimization.supportsNativeMTP)
        #expect(optimization.nativeMTPRuntimeSupported == runtimeSupported)
        #expect(optimization.detectedFeatures.contains(.nativeMTP))
        #expect(optimization.implementedFeatures.contains(.nativeMTP) == runtimeSupported)
        #expect(optimization.pendingRuntimeFeatures.contains(.nativeMTP) != runtimeSupported)
    }

    private static func makeTemporaryModelDirectory(
        config: [String: Any],
        tokenizerConfig: [String: Any]
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "MLXModelProfileMTPDetectionTests-\(UUID().uuidString)",
                isDirectory: true
            )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try writeJSON(config, to: directory.appendingPathComponent("config.json"))
        try writeJSON(tokenizerConfig, to: directory.appendingPathComponent("tokenizer_config.json"))
        return directory
    }

    private static func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        try data.write(to: url)
    }

    private static func writeSafetensorsHeader(keys: [String], to url: URL) throws {
        var header: [String: Any] = ["__metadata__": [:]]
        for (index, key) in keys.enumerated() {
            header[key] = [
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [index * 2, (index + 1) * 2]
            ]
        }
        let headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys]
        )
        var data = littleEndianUInt64Data(UInt64(headerData.count))
        data.append(headerData)
        data.append(Data(repeating: 0, count: keys.count * 2))
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
