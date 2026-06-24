import MLXNN

extension Module {
    /// Returns the logical parameter count, expanding quantized layer storage back to weight count.
    public func numParameters() -> Int {
        var total = 0

        for module in leafModules().flattenedValues() {
            total += module.logicalParameterCount
        }

        return total
    }
}

private extension Module {
    @inline(__always)
    var logicalParameterCount: Int {
        if let layer = self as? QuantizedLinear {
            return layer.scales.size * layer.groupSize
        }

        if let layer = self as? QuantizedEmbedding {
            return layer.scales.size * layer.groupSize
        }

        var total = 0
        for parameter in parameters().flattenedValues() {
            total += parameter.size
        }
        return total
    }
}
