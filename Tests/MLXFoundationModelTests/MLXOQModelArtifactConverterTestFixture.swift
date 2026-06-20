import Foundation
import MLX

struct MLXOQModelArtifactConverterTestFixture {
    let modelDirectory: URL
    let outputDirectory: URL
    let rootDirectory: URL

    static let copiedTensorName = "model.layers.0.input_layernorm.weight"
    static let quantizedTensorName = "model.layers.0.self_attn.q_proj.weight"
    static let tokenizerFilename = "tokenizer.json"
    static let visualTensorName = "visual.patch_embed.weight"

    static func make() throws -> Self {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXOQArtifactConverter-\(UUID().uuidString)")
        let modelDirectory = root.appendingPathComponent("source")
        let outputDirectory = root.appendingPathComponent("converted")
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        try writeConfig(to: modelDirectory)
        try writeTokenizer(to: modelDirectory)
        try writeWeights(to: modelDirectory)
        return Self(
            modelDirectory: modelDirectory,
            outputDirectory: outputDirectory,
            rootDirectory: root
        )
    }

    private static func writeConfig(to directory: URL) throws {
        let config: [String: Any] = [
            "model_type": "qwen2",
            "num_hidden_layers": 8
        ]
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: directory.appendingPathComponent("config.json"))
    }

    private static func writeTokenizer(to directory: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: ["version": "test"],
            options: [.sortedKeys]
        )
        try data.write(to: directory.appendingPathComponent(tokenizerFilename))
    }

    private static func writeWeights(to directory: URL) throws {
        try MLX.save(
            arrays: [
                copiedTensorName: MLXArray([Float](repeating: 1, count: 4)),
                quantizedTensorName: quantizedSource(),
                visualTensorName: MLX.ones([2, 64])
            ],
            url: directory.appendingPathComponent("model.safetensors")
        )
    }

    private static func quantizedSource() -> MLXArray {
        MLXArray((0..<128).map { Float($0) / 32 }).reshaped([2, 64])
    }
}
