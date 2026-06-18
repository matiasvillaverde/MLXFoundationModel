import Foundation

enum MLXRealModelEnvironment {
    private static let environment = ProcessInfo.processInfo.environment
    private static let tildeSlashPrefixLength = 2

    static var isEnabled: Bool {
        environment["MLX_RUN_REAL_MODEL_TESTS"] == "1"
    }

    static var scope: String {
        environment["MLX_REAL_MODEL_SCOPE"] ?? "smoke"
    }

    static var modelRoot: URL {
        if let path = environment["MLX_TEST_MODELS_DIR"], !path.isEmpty {
            return URL(fileURLWithPath: expandTilde(in: path), isDirectory: true)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(".models", isDirectory: true)
    }

    static func selectedModels(from models: [MLXRealModelCatalog.Model]) -> [MLXRealModelCatalog.Model] {
        let downloadable = models.filter(\.isDownloadable)
        switch scope {
        case "all":
            return downloadable

        case "downloaded":
            return downloadable.filter { hasModelFiles(for: $0) }

        default:
            return downloadable.filter { $0.tags.contains("smoke") }
        }
    }

    static func modelURL(for model: MLXRealModelCatalog.Model) -> URL {
        modelRoot.appendingPathComponent(model.relativePath, isDirectory: true)
    }

    static func hasModelFiles(for model: MLXRealModelCatalog.Model) -> Bool {
        hasModelFiles(at: modelURL(for: model))
    }

    static func hasModelFiles(at url: URL) -> Bool {
        let fileManager = FileManager.default
        let path = url.path
        let hasConfig = fileManager.fileExists(atPath: "\(path)/config.json")
        let hasTokenizer = fileManager.fileExists(atPath: "\(path)/tokenizer.json")
        let hasSingleFileWeights = fileManager.fileExists(atPath: "\(path)/model.safetensors")
        let hasIndexedWeights = fileManager.fileExists(atPath: "\(path)/model.safetensors.index.json")
        let hasShardWeights = (try? fileManager.contentsOfDirectory(atPath: path).contains { filename in
            filename.hasSuffix(".safetensors")
        }) ?? false

        return hasConfig && hasTokenizer && (hasSingleFileWeights || hasIndexedWeights || hasShardWeights)
    }

    static func missingModelsMessage(_ models: [MLXRealModelCatalog.Model]) -> String {
        let ids = models.map(\.id).joined(separator: ", ")
        return """
        Missing model weights for: \(ids)
        Download with:
        MLX_ASSUME_YES=1 MLX_MODEL_FILTER=\(scope) scripts/download-test-models.sh
        or set MLX_TEST_MODELS_DIR to an existing model root.
        """
    }

    private static func expandTilde(in path: String) -> String {
        if path == "~" {
            return NSHomeDirectory()
        }
        if path.hasPrefix("~/") {
            let suffix = String(path.dropFirst(tildeSlashPrefixLength))
            return URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(suffix).path
        }
        return path
    }
}
