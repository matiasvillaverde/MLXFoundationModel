import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("NanoChat architecture")
struct NanoChatArchitectureTests {
    @Test("decodes NanoChat configuration with project defaults")
    func decodesNanoChatConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "intermediate_size": 32
        }
        """#

        let config = try JSONDecoder.json5().decode(
            NanoChatConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "nanochat")
        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.logitsSoftcap == 15)
    }

    @Test("builds NanoChat attention layout and rotary plan")
    func buildsNanoChatAttentionLayoutAndRotaryPlan() {
        let config = Self.smallConfig(ropeTheta: 100)
        let layout = NanoChatAttentionLayout(config)
        let rotaryPlan = NanoChatRotaryPlan(layout: layout, theta: config.ropeTheta)
        let freqs = rotaryPlan.frequencies()
        eval(freqs)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
        #expect(rotaryPlan.halfDimensions == 2)
        #expect(freqs.shape == [2])
    }

    @Test("constructs NanoChat model with cache, adapters, and greedy fast path")
    func constructsNanoChatModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = NanoChatModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "nanochat")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["c_q", "c_v"])
    }

    @Test("tiny NanoChat model produces finite capped logits")
    func tinyNanoChatModelProducesFiniteCappedLogits() {
        let model = NanoChatModel(Self.smallConfig(hiddenLayers: 1, logitsSoftcap: 0.5))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
        #expect(logits.asArray(Float.self).allSatisfy { abs($0) <= 0.5 })
    }

    @Test("softcap leaves logits unchanged when cap is disabled")
    func softcapLeavesLogitsUnchangedWhenDisabled() {
        let input = MLXArray([Float(-2), 0, 2])
        let output = NanoChatLogitSoftcap(0).apply(to: input)
        eval(output)

        #expect(output.asArray(Float.self) == [-2, 0, 2])
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        ropeTheta: Float = 10_000,
        logitsSoftcap: Float = 15
    ) -> NanoChatConfiguration {
        NanoChatConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            attentionHeads: 4,
            kvHeads: 2,
            vocabularySize: 64,
            maxPositionEmbeddings: 64,
            intermediateSize: 32,
            ropeTheta: ropeTheta,
            logitsSoftcap: logitsSoftcap
        )
    }
}
