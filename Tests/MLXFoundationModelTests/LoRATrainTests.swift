import MLX
@testable import MLXLocalModels
import MLXNN
import Testing

@Suite("LoRA training helpers")
struct LoRATrainTests {
    @Test("batch iterator builds shifted targets and prediction lengths")
    func batchIteratorBuildsShiftedTargetsAndPredictionLengths() throws {
        var iterator = LoRABatchIterator(
            dataset: ["hello world <extra>", "ST <eos>", "hello"],
            tokenizer: PreparedGenerationTokenizer(),
            batchSize: 3,
            train: false
        )

        let nextBatch = iterator.next()
        let batch = try #require(nextBatch)
        eval(batch.0, batch.1, batch.2)

        #expect(batch.0.shape == [3, 2])
        #expect(batch.1.shape == [3, 2])
        #expect(batch.0.asArray(Int32.self) == [10, 11, 20, 99, 10, 0])
        #expect(batch.1.asArray(Int32.self) == [11, 100, 99, 0, 0, 0])
        #expect(batch.2.asArray(Int32.self) == [2, 1, 0])

        if iterator.next() != nil {
            Issue.record("Validation iterator should stop after one pass")
        }
    }

    @Test("evaluate aggregates weighted batch losses and honors batch count")
    func evaluateAggregatesWeightedBatchLossesAndHonorsBatchCount() {
        var callCount = 0
        let result = LoRATrain.evaluate(
            model: Module(),
            dataset: ["hello world", "ST <eos>", "<extra> hello"],
            loss: { _, inputs, _, lengths in
                callCount += 1
                #expect(inputs.shape == [1, 1])
                #expect(lengths.asArray(Int32.self) == [1])
                return (MLXArray(Float(callCount)), MLXArray(2))
            },
            tokenizer: PreparedGenerationTokenizer(),
            batchSize: 1,
            batchCount: 2
        )

        #expect(callCount == 2)
        #expect(abs(result - 1.5) < 0.0001)
    }

    @Test("convert replaces linear children and fuse can dequantize quantized adapters")
    func convertReplacesLinearChildrenAndFuseCanDequantizeQuantizedAdapters() throws {
        try Device.withDefaultDevice(.cpu) {
            let block = LoRATrainBlock()
            let layers: LoRALinearLayers = [(block, ["dense", "quantized"])]

            LoRATrain.convert(model: block, layers: layers)

            let dense = try #require(Self.child("dense", in: block) as? LoRALinear)
            let quantized = try #require(Self.child("quantized", in: block) as? QLoRALinear)

            #expect(Self.trainableKeys(dense) == ["lora_a", "lora_b"])
            #expect(Self.trainableKeys(quantized) == ["lora_a", "lora_b"])

            LoRATrain.fuse(model: block, layers: layers, deQuantize: true)

            let fusedDense = try #require(Self.child("dense", in: block) as? Linear)
            let fusedQuantized = try #require(Self.child("quantized", in: block) as? Linear)

            #expect(!(fusedDense is LoRALayer))
            #expect(!(fusedQuantized is LoRALayer))
            #expect(!(fusedQuantized is QuantizedLinear))
        }
    }

    private static func child(_ key: String, in module: Module) -> Module? {
        guard let item = module.children()[key], case .value(let child) = item else {
            return nil
        }
        return child
    }

    private static func trainableKeys(_ module: Module) -> Set<String> {
        Set(module.trainableParameters().flattened().map(\.0))
    }

    private final class LoRATrainBlock: Module {
        @ModuleInfo var dense: Linear
        @ModuleInfo var quantized: Linear

        override init() {
            dense = Linear(
                weight: MLXArray((0 ..< 6).map { Float($0) / 10 }).reshaped([2, 3]),
                bias: MLXArray([Float(0), Float(1)])
            )
            quantized = QuantizedLinear(
                weight: MLXArray((0 ..< 64).map { Float($0) / 64 }).reshaped([2, 32]),
                bias: MLXArray([Float(0), Float(1)]),
                groupSize: 32,
                bits: 4,
                mode: .affine
            )
            super.init()
        }

        deinit {
            // Required by the strict test lint profile for reference types.
        }
    }
}
