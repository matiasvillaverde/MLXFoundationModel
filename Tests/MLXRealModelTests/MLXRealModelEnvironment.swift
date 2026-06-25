import Foundation

enum MLXRealModelEnvironment {
    private static let environment = ProcessInfo.processInfo.environment
    private static let oneGiB: Int64 = 1_073_741_824
    private static let modelLoadOverheadMultiplier = 2.5
    private static let largeHostReserveGB = 8
    private static let smallHostReserveGB = 4
    private static let tildeSlashPrefixLength = 2
    private static let modelArtifactExtensions: Set<String> = [
        "bin",
        "mlx",
        "npz",
        "safetensors"
    ]

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

    static var architectureGenerationTokenLimit: Int {
        integerValue(for: "MLX_REAL_MODEL_GENERATION_TOKENS", defaultValue: 2, minimumValue: 1)
    }

    static var architectureGenerationTimeoutSeconds: Int {
        integerValue(for: "MLX_REAL_MODEL_GENERATION_TIMEOUT_SECONDS", defaultValue: 120, minimumValue: 1)
    }

    static var stressIterationCount: Int {
        integerValue(for: "MLX_REAL_MODEL_STRESS_ITERATIONS", defaultValue: 3, minimumValue: 1)
    }

    static var stressGenerationTokenLimit: Int {
        integerValue(for: "MLX_REAL_MODEL_STRESS_TOKENS", defaultValue: 32, minimumValue: 1)
    }

    static var stressTimeoutSeconds: Int {
        integerValue(for: "MLX_REAL_MODEL_STRESS_TIMEOUT_SECONDS", defaultValue: 240, minimumValue: 1)
    }

    static func selectedModels(from models: [MLXRealModelCatalog.Model]) -> [MLXRealModelCatalog.Model] {
        let downloadable = models.filter(\.isDownloadable)
        let selectedIDs = selectedModelIDs
        let selected: [MLXRealModelCatalog.Model]
        if !selectedIDs.isEmpty {
            selected = downloadable.filter { selectedIDs.contains($0.id) }
        } else {
            switch scope {
            case "all":
                selected = downloadable

            case "relevant":
                selected = downloadable.filter { $0.tags.contains("relevant") }

            case "main":
                selected = downloadable.filter { $0.tags.contains("main") }

            case "downloaded":
                selected = downloadable.filter { hasModelFiles(for: $0) }

            default:
                selected = downloadable.filter { matches(scope, model: $0) }
            }
        }
        guard environment["MLX_ALLOW_OVERSIZED_MODELS"] != "1" else {
            return selected
        }
        return selected.filter { model in
            canRunWithinHostMemory(
                model,
                estimatedModelLoadBytes: estimatedModelLoadBytes(for: model),
                hostMemoryGB: hostMemoryGB
            )
        }
    }

    static func canRunModel(id: String) -> Bool {
        guard environment["MLX_ALLOW_OVERSIZED_MODELS"] != "1",
            let model = try? MLXRealModelCatalog.load().first(where: { $0.id == id })
        else {
            return true
        }
        return canRunWithinHostMemory(
            model,
            estimatedModelLoadBytes: estimatedModelLoadBytes(for: model),
            hostMemoryGB: hostMemoryGB
        )
    }

    static func canRunWithinHostMemory(
        _ model: MLXRealModelCatalog.Model,
        estimatedModelLoadBytes: Int64?,
        hostMemoryGB: Int
    ) -> Bool {
        if let minimumMemoryGB = model.minimumMemoryGB {
            return minimumMemoryGB <= hostMemoryGB
        }
        guard let estimatedModelLoadBytes else {
            return true
        }
        return estimatedRuntimeBytes(forModelLoadBytes: estimatedModelLoadBytes) <=
            hostModelBudgetBytes(hostMemoryGB: hostMemoryGB)
    }

    static func estimatedRuntimeMemoryGB(forModelLoadBytes bytes: Int64) -> Int {
        let runtimeBytes = estimatedRuntimeBytes(forModelLoadBytes: bytes)
        return Int((Double(runtimeBytes) / Double(oneGiB)).rounded(.up))
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

    private static func matches(_ filter: String, model: MLXRealModelCatalog.Model) -> Bool {
        let normalizedFilter = filter.lowercased()
        guard !normalizedFilter.isEmpty else {
            return false
        }
        return [
            model.id,
            model.displayName,
            model.architecture,
            model.repository ?? "",
            model.relativePath,
            model.tags.joined(separator: ",")
        ]
        .joined(separator: " ")
        .lowercased()
        .contains(normalizedFilter)
    }

    static func hasModelFiles(for model: MLXRealModelCatalog.Model) -> Bool {
        hasModelFiles(at: modelURL(for: model))
    }

    static func hasModelFiles(at url: URL) -> Bool {
        let fileManager = FileManager.default
        let path = url.path
        let hasConfig = fileManager.fileExists(atPath: "\(path)/config.json")
        let hasTokenizerJSON = fileManager.fileExists(atPath: "\(path)/tokenizer.json")
        let hasSentencePieceTokenizer = fileManager.fileExists(atPath: "\(path)/tokenizer.model")
        let hasTokenizer = hasTokenizerJSON || hasSentencePieceTokenizer
        let hasSingleFileWeights = fileManager.fileExists(atPath: "\(path)/model.safetensors")
        let hasIndexedWeights = fileManager.fileExists(atPath: "\(path)/model.safetensors.index.json")
        let hasShardWeights = (try? fileManager.contentsOfDirectory(atPath: path).contains { filename in
            filename.hasSuffix(".safetensors")
        }) ?? false

        return hasConfig && hasTokenizer && (hasSingleFileWeights || hasIndexedWeights || hasShardWeights)
    }

    static func estimatedModelLoadBytes(for model: MLXRealModelCatalog.Model) -> Int64? {
        estimatedModelLoadBytes(at: modelURL(for: model))
    }

    static func estimatedModelLoadBytes(at url: URL) -> Int64? {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return nil
        }

        var byteCount: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let bytes = modelArtifactByteCount(for: fileURL) else {
                continue
            }
            byteCount += bytes
        }
        return byteCount > 0 ? byteCount : nil
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

        case "main", "relevant", "smoke":
            return "MLX_ASSUME_YES=1 MLX_MODEL_FILTER=\(scope) scripts/download-test-models.sh"

        default:
            return "MLX_ASSUME_YES=1 MLX_MODEL_FILTER='\(scope)' scripts/download-test-models.sh"
        }
    }

    private static var hostMemoryGB: Int {
        if let value = environment["MLX_HOST_MEMORY_GB"], let integer = Int(value) {
            return integer
        }
        return max(1, Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824))
    }

    private static func estimatedRuntimeBytes(forModelLoadBytes bytes: Int64) -> Int64 {
        guard bytes > 0 else {
            return 0
        }
        return Int64((Double(bytes) * modelLoadOverheadMultiplier).rounded(.up))
    }

    private static func hostModelBudgetBytes(hostMemoryGB: Int) -> Int64 {
        let reserve = hostMemoryGB < 24 ? smallHostReserveGB : largeHostReserveGB
        return Int64(max(1, hostMemoryGB - reserve)) * oneGiB
    }

    private static func modelArtifactByteCount(for url: URL) -> Int64? {
        guard modelArtifactExtensions.contains(url.pathExtension.lowercased()),
            let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
            values.isRegularFile == true,
            let fileSize = values.fileSize,
            fileSize > 0
        else {
            return nil
        }
        return Int64(fileSize)
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

    private static func integerValue(
        for key: String,
        defaultValue: Int,
        minimumValue: Int
    ) -> Int {
        guard
            let value = environment[key],
            let integer = Int(value)
        else {
            return defaultValue
        }
        return max(integer, minimumValue)
    }
}
