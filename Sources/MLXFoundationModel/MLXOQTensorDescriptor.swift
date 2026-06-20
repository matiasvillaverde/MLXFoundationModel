import Foundation

/// Minimal tensor metadata needed to plan oQ quantization without loading weights.
public struct MLXOQTensorDescriptor: Equatable, Hashable, Sendable {
    public let name: String
    public let shape: [Int]

    public init(name: String, shape: [Int]) {
        self.name = name
        self.shape = shape
    }
}
