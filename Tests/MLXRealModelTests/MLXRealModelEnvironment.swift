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
        let selectedIDs = selectedModelIDs
        if !selectedIDs.isEmpty {
            return downloadable.filter { selectedIDs.contains($0.id) }
        }
        switch scope {
        case "all":
            return downloadable

        case "main":
            return downloadable.filter { $0.tags.contains("main") }

        case "downloaded":
            return downloadable.filter { hasModelFiles(for: $0) }

        default:
            return downloadable.filter { $0.tags.contains("smoke") }
        }
    }

    private static var selectedModelIDs: Set<String> {
        guard let value = environment["MLX_REAL_MODEL_IDS"], !value.isEmpty else {
            return []
        }
        return Set(
            value
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        )
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
        \(downloadCommand)
        or set MLX_TEST_MODELS_DIR to an existing model root.
        """
    }

    private static var downloadCommand: String {
        switch scope {
        case "all", "downloaded":
            return "MLX_ASSUME_YES=1 scripts/download-test-models.sh"

        default:
            return "MLX_ASSUME_YES=1 MLX_MODEL_FILTER=\(scope) scripts/download-test-models.sh"
        }
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
