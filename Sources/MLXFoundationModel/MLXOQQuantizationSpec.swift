import Foundation

/// Quantization parameters selected for a tensor.
public struct MLXOQQuantizationSpec: Codable, Equatable, Hashable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    public init(bits: Int, groupSize: Int, mode: String = "affine") {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
    }
}
