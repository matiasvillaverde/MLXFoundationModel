enum ExaoneMoETestFixtures {
    static let configJSON = #"""
    {
        "model_type": "exaone_moe",
        "architectures": ["ExaoneMoeForCausalLM"],
        "vocab_size": 153600,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 512,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_shared_experts": 1,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 131072,
        "sliding_window": 128,
        "sliding_window_pattern": "LLLG",
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
        ],
        "mlp_layer_types": [
            "dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
            "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"
        ],
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "tie_word_embeddings": false
    }
    """#
}
