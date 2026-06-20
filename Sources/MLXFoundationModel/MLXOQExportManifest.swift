import Foundation

/// Header-only conversion manifest for exporting an MLX model with oQ decisions.
public struct MLXOQExportManifest: Codable, Equatable, Sendable {
    public let baselineBitsPerWeight: Double
    public let effectiveBitsPerWeight: Double
    public let entries: [MLXOQExportTensorEntry]
    public let estimatedSerializedBytes: Int
    public let level: String

    public init(
        level: String,
        entries: [MLXOQExportTensorEntry],
        baselineBitsPerWeight: Double,
        effectiveBitsPerWeight: Double,
        estimatedSerializedBytes: Int
    ) {
        self.baselineBitsPerWeight = baselineBitsPerWeight
        self.effectiveBitsPerWeight = effectiveBitsPerWeight
        self.entries = entries
        self.estimatedSerializedBytes = estimatedSerializedBytes
        self.level = level
    }

    public var copiedTensorCount: Int {
        entries.filter { !$0.isQuantized }.count
    }

    public var outputTensorNames: [String] {
        entries.flatMap(\.outputNames)
    }

    public var quantizedTensorCount: Int {
        entries.filter(\.isQuantized).count
    }

    public var sidecarTensorCount: Int {
        entries.reduce(0) { total, entry in
            total + max(entry.outputNames.count - 1, 0)
        }
    }
}
