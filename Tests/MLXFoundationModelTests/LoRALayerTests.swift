import MLX
@testable import MLXLocalModels
import MLXNN
import Testing

@Suite("LoRA layers")
struct LoRALayerTests {
    @Test("dense conversion keeps base weights and trains only adapters")
    func denseConversionKeepsBaseWeightsAndTrainsOnlyAdapters() throws {
        let base = Self.linear(inputDimensions: 3, outputDimensions: 2)
        let layer = try #require(
            LoRALinear.from(linear: base, rank: 2, scale: 4) as? LoRALinear
        )

        #expect(layer.scale == 4)
        #expect(layer.shape.0 == 2)
        #expect(layer.shape.1 == 3)
        #expect(layer.loraA.shape == [3, 2])
        #expect(layer.loraB.shape == [2, 2])
        #expect(layer.weight.shape == base.weight.shape)
        #expect(layer.bias?.shape == base.bias?.shape)
        #expect(Self.trainableKeys(layer) == ["lora_a", "lora_b"])
        #expect(layer.noGrad().contains("weight"))
        #expect(layer.noGrad().contains("bias"))
        #expect(!layer.noGrad().contains("lora_a"))
        #expect(!layer.noGrad().contains("lora_b"))
    }

    @Test("dense adapter starts as no-op and fuses back to base weight")
    func denseAdapterStartsAsNoOpAndFusesBackToBaseWeight() throws {
        try Device.withDefaultDevice(.cpu) {
            let base = Self.linear(inputDimensions: 3, outputDimensions: 2)
            let layer = try #require(
                LoRALinear.from(linear: base, rank: 2, scale: 4) as? LoRALinear
            )
            let input = MLXArray([Float(1), Float(0), Float(-1)]).reshaped([1, 3])
            let baseOutput = base(input)
            let loraOutput = layer(input)
            let fused = try #require(layer.fused() as? Linear)

            eval(baseOutput, loraOutput, fused.weight, base.weight)
            #expect(loraOutput.asArray(Float.self) == baseOutput.asArray(Float.self))
            #expect(fused.weight.asArray(Float.self) == base.weight.asArray(Float.self))
        }
    }

    @Test("quantized conversion preserves quantization metadata and trains only adapters")
    func quantizedConversionPreservesMetadataAndTrainsOnlyAdapters() throws {
        try Device.withDefaultDevice(.cpu) {
            let base = QuantizedLinear(
                weight: MLXArray((0 ..< 64).map { Float($0) / 64 }).reshaped([2, 32]),
                bias: MLXArray([Float(0), Float(1)]),
                groupSize: 32,
                bits: 4,
                mode: .affine
            )
            let layer = try #require(
                LoRALinear.from(linear: base, rank: 3, scale: 5) as? QLoRALinear
            )
            let reverted = try #require(layer.reverted() as? QuantizedLinear)

            #expect(layer.scale == 5)
            #expect(layer.groupSize == 32)
            #expect(layer.bits == 4)
            #expect(layer.mode == .affine)
            #expect(layer.loraA.shape == [32, 3])
            #expect(layer.loraB.shape == [3, 2])
            #expect(Self.trainableKeys(layer) == ["lora_a", "lora_b"])
            #expect(layer.noGrad().contains("weight"))
            #expect(layer.noGrad().contains("scales"))
            #expect(!layer.noGrad().contains("lora_a"))
            #expect(!layer.noGrad().contains("lora_b"))
            #expect(reverted.groupSize == base.groupSize)
            #expect(reverted.bits == base.bits)
            #expect(reverted.mode == base.mode)
        }
    }

    private static func linear(inputDimensions: Int, outputDimensions: Int) -> Linear {
        Linear(
            weight: MLXArray((0 ..< outputDimensions * inputDimensions).map { Float($0) / 10 })
                .reshaped([outputDimensions, inputDimensions]),
            bias: MLXArray((0 ..< outputDimensions).map(Float.init))
        )
    }

    private static func trainableKeys(_ module: Module) -> Set<String> {
        Set(module.trainableParameters().flattened().map(\.0))
    }
}
