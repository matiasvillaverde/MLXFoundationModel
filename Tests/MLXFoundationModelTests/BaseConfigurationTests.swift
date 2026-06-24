import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Base model configuration")
struct BaseConfigurationTests {
    @Test("decodes model type and EOS token ids")
    func decodesModelTypeAndEOSTokens() throws {
        let single = try decode(#"{"model_type":"qwen3","eos_token_id":151645}"#)
        let multiple = try decode(#"{"model_type":"qwen3","eos_token_id":[151645,151643]}"#)

        #expect(single.modelType == "qwen3")
        #expect(single.eosTokenIds?.values == [151_645])
        #expect(multiple.eosTokenIds?.values == [151_645, 151_643])
    }

    @Test("decodes quantization and per-layer overrides")
    func decodesQuantizationOverrides() throws {
        let configuration = try decode(Self.quantizationOverridesJSON)

        let quantization = try #require(configuration.quantizationContainer?.quantization)
        let perLayer = try #require(configuration.perLayerQuantization)

        assertDefaultQuantization(quantization)
        try assertLayerOverrides(perLayer, defaultQuantization: quantization)
    }

    private func assertDefaultQuantization(_ quantization: BaseConfiguration.Quantization) {
        #expect(quantization.groupSize == 64)
        #expect(quantization.bits == 4)
        #expect(quantization.quantMethod == "gptq")
        #expect(quantization.linearClass == "QuantizedLinear")
        #expect(quantization.mode == .affine)
    }

    private func assertLayerOverrides(
        _ perLayer: BaseConfiguration.PerLayerQuantization,
        defaultQuantization: BaseConfiguration.Quantization
    ) throws {
        #expect(
            perLayer.quantization(layer: "model.layers.2.mlp.down_proj") == defaultQuantization
        )
        #expect(perLayer.quantization(layer: "model.embed_tokens") == nil)
        let override = try #require(
            perLayer.quantization(layer: "model.layers.0.self_attn.q_norm")
        )
        #expect(override.groupSize == 32)
        #expect(override.bits == 8)
        #expect(override.mode == .affine)
        #expect(
            perLayer.quantization(layer: "model.layers.1.self_attn.q_norm") == defaultQuantization
        )
    }

    @Test("falls back to affine when quantization mode is absent or unknown")
    func fallsBackToAffineMode() throws {
        let absent = try decode(#"{"model_type":"qwen3","quantization":{"group_size":64,"bits":4}}"#)
        let unknown = try decode(
            #"{"model_type":"qwen3","quantization":{"group_size":64,"bits":4,"mode":"unknown"}}"#
        )

        #expect(absent.quantizationContainer?.quantization.mode == .affine)
        #expect(unknown.quantizationContainer?.quantization.mode == .affine)
    }

    @Test("encodes quantization overrides")
    func encodesQuantizationOverrides() throws {
        let configuration = try decode(Self.encodableQuantizationOverridesJSON)

        let encoded = try Self.encoder.encode(configuration)
        let object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        let quantization = try #require(object["quantization"] as? [String: Any])
        let override = try #require(
            quantization["model.layers.0.self_attn.q_norm"] as? [String: Any]
        )

        #expect(quantization["group_size"] as? Int == 64)
        #expect(quantization["bits"] as? Int == 4)
        #expect(quantization["model.embed_tokens"] as? Bool == false)
        #expect(override["group_size"] as? Int == 32)
        #expect(override["bits"] as? Int == 8)
    }

    private static let decoder = JSONDecoder()
    private static let encoder: JSONEncoder = {
        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = [.sortedKeys]
        return jsonEncoder
    }()

    private static let quantizationOverridesJSON = [
        #"{"model_type":"qwen3","#,
        #""quantization":{"group_size":64,"bits":4,"#,
        #""quant_method":"gptq","linear_class":"QuantizedLinear","mode":"affine","#,
        #""model.embed_tokens":false,"#,
        #""model.layers.0.self_attn.q_norm":{"group_size":32,"bits":8,"#,
        #""quantization_mode":"affine"},"#,
        #""model.layers.1.self_attn.q_norm":true}}"#
    ].joined()

    private static let encodableQuantizationOverridesJSON = [
        #"{"model_type":"qwen3","#,
        #""quantization":{"group_size":64,"bits":4,"#,
        #""model.embed_tokens":false,"#,
        #""model.layers.0.self_attn.q_norm":{"group_size":32,"bits":8}}}"#
    ].joined()

    private func decode(_ json: String) throws -> BaseConfiguration {
        try Self.decoder.decode(BaseConfiguration.self, from: Data(json.utf8))
    }
}
