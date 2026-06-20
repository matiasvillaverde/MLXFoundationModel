import Foundation

/// Quantization metadata inferred from an MLX model configuration.
public struct MLXModelQuantizationProfile: Codable, Equatable, Hashable, Sendable {
    public let bits: Int?
    public let groupSize: Int?
    public let method: String?
    public let linearClass: String?
    public let mode: String?
    public let format: String?

    public init(
        bits: Int? = nil,
        groupSize: Int? = nil,
        method: String? = nil,
        linearClass: String? = nil,
        mode: String? = nil,
        format: String? = nil
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.method = method
        self.linearClass = linearClass
        self.mode = mode
        self.format = format
    }
}
