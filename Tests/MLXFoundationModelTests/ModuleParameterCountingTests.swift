@testable import MLXLocalModels
import MLXNN
import Testing

@Suite("Module parameter counting")
struct ModuleParameterCountingTests {
    @Test("counts dense module parameters without intermediate collections")
    func countsDenseModuleParameters() {
        let model = DenseModule()

        #expect(model.numParameters() == 26)
    }

    @Test("counts quantized module parameters as logical uncompressed weights")
    func countsQuantizedModuleParameters() {
        let model = QuantizedModule()
        let expectedCount =
            model.projection.scales.size * model.projection.groupSize
            + model.embedding.scales.size * model.embedding.groupSize

        #expect(model.numParameters() == expectedCount)
    }

    private final class DenseModule: Module {
        @ModuleInfo(key: "projection")
        var projection: Linear
        @ModuleInfo(key: "embedding")
        var embedding: Embedding

        override init() {
            _projection.wrappedValue = Linear(3, 4, bias: true)
            _embedding.wrappedValue = Embedding(embeddingCount: 5, dimensions: 2)
        }

        deinit {
            // Required by the strict test lint profile.
        }
    }

    private final class QuantizedModule: Module {
        @ModuleInfo(key: "projection")
        var projection: QuantizedLinear
        @ModuleInfo(key: "embedding")
        var embedding: QuantizedEmbedding

        override init() {
            _projection.wrappedValue = QuantizedLinear(32, 3, bias: false, groupSize: 32, bits: 4)
            _embedding.wrappedValue = QuantizedEmbedding(
                embeddingCount: 5,
                dimensions: 32,
                groupSize: 32,
                bits: 4
            )
        }

        deinit {
            // Required by the strict test lint profile.
        }
    }
}
