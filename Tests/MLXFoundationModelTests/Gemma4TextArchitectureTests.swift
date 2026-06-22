import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Gemma 4 text architecture")
struct Gemma4TextArchitectureTests {
    @Test("sanitizer drops unused shared-KV tail weights")
    func sanitizerDropsUnusedSharedKVTailWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4Model(try Self.decodeConfig(numKVSharedLayers: 2))
            let weight = MLXArray.zeros([1])

            let sanitized = model.sanitize(weights: [
                "language_model.model.layers.1.self_attn.k_proj.weight": weight,
                "language_model.model.layers.2.self_attn.k_proj.weight": weight,
                "language_model.model.layers.2.self_attn.v_proj.scales": weight,
                "language_model.model.layers.3.self_attn.k_norm.weight": weight,
                "language_model.model.layers.3.self_attn.q_proj.weight": weight,
                "language_model.model.layers.3.self_attn.o_proj.weight": weight
            ])

            #expect(sanitized["language_model.model.layers.1.self_attn.k_proj.weight"] != nil)
            #expect(sanitized["language_model.model.layers.2.self_attn.k_proj.weight"] == nil)
            #expect(sanitized["language_model.model.layers.2.self_attn.v_proj.scales"] == nil)
            #expect(sanitized["language_model.model.layers.3.self_attn.k_norm.weight"] == nil)
            #expect(sanitized["language_model.model.layers.3.self_attn.q_proj.weight"] != nil)
            #expect(sanitized["language_model.model.layers.3.self_attn.o_proj.weight"] != nil)
        }
    }

    @Test("sanitizer keeps KV weights when no shared-KV tail exists")
    func sanitizerKeepsKVWeightsWithoutSharedKVTail() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4Model(try Self.decodeConfig(numKVSharedLayers: 0))
            let weight = MLXArray.zeros([1])

            let sanitized = model.sanitize(weights: [
                "language_model.model.layers.3.self_attn.k_proj.weight": weight,
                "language_model.model.layers.3.self_attn.v_proj.weight": weight,
                "language_model.model.layers.3.self_attn.k_norm.weight": weight
            ])

            #expect(sanitized["language_model.model.layers.3.self_attn.k_proj.weight"] != nil)
            #expect(sanitized["language_model.model.layers.3.self_attn.v_proj.weight"] != nil)
            #expect(sanitized["language_model.model.layers.3.self_attn.k_norm.weight"] != nil)
        }
    }

    // swiftlint:disable function_body_length indentation_width
    private static func decodeConfig(numKVSharedLayers: Int) throws -> Gemma4TextConfiguration {
        let json = """
        {
          "model_type": "gemma4_text",
          "hidden_size": 16,
          "num_hidden_layers": 4,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "head_dim": 8,
          "rms_norm_eps": 0.000001,
          "vocab_size": 64,
          "num_key_value_heads": 1,
          "num_kv_shared_layers": \(numKVSharedLayers),
          "hidden_size_per_layer_input": 0,
          "sliding_window": 16,
          "max_position_embeddings": 128,
          "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
          ],
          "rope_parameters": {
            "full_attention": {
              "rope_theta": 1000000.0,
              "rope_type": "proportional",
              "partial_rotary_factor": 0.25
            },
            "sliding_attention": {
              "rope_theta": 10000.0,
              "rope_type": "default"
            }
          },
          "rope_traditional": false,
          "attention_k_eq_v": false,
          "use_double_wide_mlp": true,
          "enable_moe_block": false,
          "use_second_mlp_block": false,
          "sliding_window_pattern": 2,
          "tie_word_embeddings": true
        }
        """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(json.utf8)
        )
    }
    // swiftlint:enable function_body_length indentation_width
}
