import Foundation

enum MLXOQLayerIndexParser {
    static func layerIndex(in path: String) -> Int? {
        guard let range = path.range(of: "layers.") else {
            return nil
        }
        var current = range.upperBound
        let start = current
        while current < path.endIndex,
            path[current].isNumber {
            current = path.index(after: current)
        }
        guard start < current else {
            return nil
        }
        return Int(path[start..<current])
    }
}
