enum HunyuanTestFixtures {
    static let hunyuan7BConfigJSON = """
    {
        "architectures": ["HunYuanForCausalLM"],
        "attention_bias": false,
        "attention_head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 4096,
        "mlp_bias": false,
        "model_type": "hunyuan",
        "moe_topk": [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ],
        "num_attention_heads": 32,
        "num_experts": 1,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_shared_expert": [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ],
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "alpha": 1000.0,
            "factor": 1.0,
            "type": "dynamic"
        },
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "use_cla": false,
        "use_mixed_mlp_moe": false,
        "use_qk_norm": true,
        "vocab_size": 129024
    }
    """

    static let hunyuanA13BMoEConfigJSON = """
    {
        "architectures": ["HunYuanMoEV1ForCausalLM"],
        "attention_bias": false,
        "attention_head_dim": 128,
        "hidden_size": 4096,
        "intermediate_size": 3072,
        "model_type": "hunyuan",
        "moe_intermediate_size": [
            3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072,
            3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072,
            3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072,
            3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072
        ],
        "moe_topk": [
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
        ],
        "num_attention_heads": 32,
        "num_experts": 64,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_shared_expert": [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ],
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "alpha": 1000.0,
            "factor": 1.0,
            "type": "dynamic"
        },
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "use_cla": false,
        "use_mixed_mlp_moe": true,
        "use_qk_norm": true,
        "vocab_size": 128167
    }
    """
}
