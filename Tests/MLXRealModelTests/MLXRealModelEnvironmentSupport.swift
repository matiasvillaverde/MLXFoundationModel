import Foundation
@testable import MLXLocalModels

enum MLXRealModelEnvironmentSupport {
    static let oneGiB: Int64 = 1_073_741_824
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

    static func hostMemoryGB(environment: [String: String]) -> Int {
        if let value = environment["MLX_HOST_MEMORY_GB"], let integer = Int(value) {
            return integer
        }
        return max(1, Int(Int64(ProcessInfo.processInfo.physicalMemory) / oneGiB))
    }

    static func memoryGuardTier(
        environment: [String: String],
        model: MLXRealModelCatalog.Model
    ) -> MLXMemoryGuardTier {
        if let value = environment["MLX_REAL_MODEL_MEMORY_GUARD_TIER"],
           let tier = MLXMemoryGuardTier(rawValue: value) {
            return tier
        }
        if let value = model.memoryGuardTier,
           let tier = MLXMemoryGuardTier(rawValue: value) {
            return tier
        }
        return .balanced
    }

    static func memoryGuardHardLimitFraction(environment: [String: String]) -> Double {
        guard let value = environment["MLX_REAL_MODEL_MEMORY_GUARD_HARD_LIMIT_FRACTION"],
              let fraction = Double(value)
        else {
            return 0.95
        }
        return fraction
    }

    static func estimatedRuntimeBytes(forModelLoadBytes bytes: Int64) -> Int64 {
        guard bytes > 0 else {
            return 0
        }
        return Int64((Double(bytes) * modelLoadOverheadMultiplier).rounded(.up))
    }

    static func hostModelBudgetBytes(hostMemoryGB: Int) -> Int64 {
        let reserve = hostMemoryGB < 24 ? smallHostReserveGB : largeHostReserveGB
        return Int64(max(1, hostMemoryGB - reserve)) * oneGiB
    }

    static func modelArtifactByteCount(for url: URL) -> Int64? {
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

    static func expandTilde(in path: String) -> String {
        if path == "~" {
            return NSHomeDirectory()
        }
        if path.hasPrefix("~/") {
            let suffix = String(path.dropFirst(tildeSlashPrefixLength))
            return URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(suffix).path
        }
        return path
    }

    static func integerValue(
        environment: [String: String],
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
