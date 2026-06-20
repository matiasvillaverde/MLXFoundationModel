import Foundation
import MLXFoundationModel

struct MLXOQExportManifestTestFixture {
    let copied = Self.copiedName
    let mxfp8 = Self.mxfp8Name
    let quantized = Self.quantizedName
    let visual = Self.visualName

    let directory: URL
    let manifest: MLXOQExportManifest

    var entriesBySourceName: [String: MLXOQExportTensorEntry] {
        Dictionary(uniqueKeysWithValues: manifest.entries.map { ($0.sourceName, $0) })
    }

    static func make() throws -> Self {
        let fixtureDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXOQExportManifest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDirectory, withIntermediateDirectories: true)
        try writeModelFiles(to: fixtureDirectory)
        let manifest = try MLXOQModelArtifactPlanner.exportManifest(
            modelDirectory: fixtureDirectory,
            level: "oQ4",
            options: .init(
                hardCapBitsPerWeight: 8,
                fixedOverrides: [
                    mxfp8Name: .init(bits: 8, groupSize: 32, mode: "mxfp8")
                ]
            )
        )
        return Self(directory: fixtureDirectory, manifest: manifest)
    }

    private init(directory: URL, manifest: MLXOQExportManifest) {
        self.directory = directory
        self.manifest = manifest
    }

    private static func writeModelFiles(to directory: URL) throws {
        try Self.writeJSON(
            ["num_hidden_layers": 8],
            to: directory.appendingPathComponent("config.json")
        )
        try Self.writeSafetensorsHeader(
            tensors: [
                copiedName: [4_096],
                mxfp8Name: [4_096, 4_096],
                quantizedName: [4_096, 4_096],
                visualName: [4_096, 4_096]
            ],
            to: directory.appendingPathComponent("model.safetensors")
        )
    }

    private static let copiedName = "model.layers.0.input_layernorm.weight"
    private static let mxfp8Name = "model.layers.1.mlp.down_proj.weight"
    private static let quantizedName = "model.layers.0.self_attn.q_proj.weight"
    private static let visualName = "visual.patch_embed.weight"

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
