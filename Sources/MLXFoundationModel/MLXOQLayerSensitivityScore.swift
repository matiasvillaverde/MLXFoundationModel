import Foundation

/// A scalar oQ sensitivity value for one model layer.
public struct MLXOQLayerSensitivityScore: Codable, Equatable, Sendable {
    public let layerIndex: Int
    public let score: Double
    public let tensorName: String

    public init(layerIndex: Int, score: Double, tensorName: String) {
        self.layerIndex = layerIndex
        self.score = score
        self.tensorName = tensorName
    }
}
