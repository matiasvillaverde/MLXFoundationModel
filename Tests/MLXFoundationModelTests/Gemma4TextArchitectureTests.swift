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

    @Test("speculative target output captures hidden states and shared KV")
    func speculativeTargetOutputCapturesHiddenStatesAndSharedKV() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4Model(try Self.decodeConfig(numKVSharedLayers: 2))
            let cache = model.newCache(parameters: nil)

            let output = model.speculativeTargetOutput(
                .init(tokens: MLXArray([1, 2])),
                cache: cache,
                state: nil
            )
            let draftHidden = model.speculativeDraftHidden(output.hiddenStates)
            let sliding = try #require(output.sharedKVStates["sliding_attention"])
            let full = try #require(output.sharedKVStates["full_attention"])
            eval(output.logits, output.hiddenStates, draftHidden, sliding.keys, full.keys)

            #expect(output.logits.shape == [1, 2, 64])
            #expect(output.hiddenStates.shape == [1, 2, 16])
            #expect(draftHidden.shape == output.hiddenStates.shape)
            #expect(Set(output.sharedKVStates.keys) == ["sliding_attention", "full_attention"])
            #expect(sliding.keys.shape[sliding.keys.shape.count - 2] == 2)
            #expect(sliding.values.shape[sliding.values.shape.count - 2] == 2)
            #expect(full.keys.shape[full.keys.shape.count - 2] == 2)
            #expect(full.values.shape[full.values.shape.count - 2] == 2)
            #expect(cache.map(\.offset) == [2, 2])
        }
    }

    // swiftlint:disable function_body_length closure_body_length
    @Test("assistant model drafts from shared KV state")
    func assistantModelDraftsFromSharedKVState() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Gemma4AssistantModel(try Self.decodeAssistantConfig())
            let sharedKVStates = [
                "sliding_attention": SharedKVState(
                    keys: MLXArray.ones([1, 1, 1, 4]),
                    values: MLXArray.ones([1, 1, 1, 4]),
                    offset: 1
                ),
                "full_attention": SharedKVState(
                    keys: MLXArray.ones([1, 1, 1, 4]),
                    values: MLXArray.ones([1, 1, 1, 4]),
                    offset: 1
                )
            ]

            let output = model.sharedKVDraftOutput(
                tokenEmbeddings: MLXArray.ones([1, 1, 16]),
                hiddenStates: MLXArray.ones([1, 1, 16]),
                sharedKVStates: sharedKVStates,
                position: 0
            )
            let greedy = try #require(model.sharedKVGreedyDraftOutput(
                tokenEmbeddings: MLXArray.ones([1, 1, 16]),
                hiddenStates: MLXArray.ones([1, 1, 16]),
                sharedKVStates: sharedKVStates,
                position: 0
            ))
            eval(output.logits, output.hiddenStates, greedy.token, greedy.hiddenStates)

            #expect(model.speculativeDraftBlockSize == 4)
            #expect(output.logits.shape == [1, 1, 64])
            #expect(output.hiddenStates.shape == [1, 1, 16])
            #expect(greedy.token.shape == [1, 1])
            #expect(greedy.hiddenStates.shape == output.hiddenStates.shape)
        }
    }
    // swiftlint:enable function_body_length closure_body_length

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

    private static func decodeAssistantConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 16,
          "use_ordered_embeddings": true,
          "num_centroids": 4,
          "centroid_intermediate_top_k": 2,
          "tie_word_embeddings": true,
          "block_size": 4,
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "head_dim": 4,
            "global_head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 0,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "final_logit_softcapping": null,
            "layer_types": ["sliding_attention", "full_attention"],
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
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "use_second_mlp_block": false,
            "num_experts": null,
            "top_k_experts": null,
            "moe_intermediate_size": null,
            "sliding_window_pattern": 2,
            "tie_word_embeddings": true
          }
        }
        """
        return try JSONDecoder().decode(
            Gemma4AssistantConfiguration.self,
            from: Data(json.utf8)
        )
    }
    // swiftlint:enable function_body_length indentation_width
}
