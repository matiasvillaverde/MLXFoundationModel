import Foundation
import MLX
@testable import MLXLocalModels

enum GLMMoEDSATestSupport {
    static func decodeConfig(indexSchedule: String) throws -> GLM4MoELiteConfiguration {
        try decodeConfig(
            indexSchedule: indexSchedule,
            nRoutedExperts: 2,
            nGroup: 1,
            topkGroup: 1,
            tieWordEmbeddings: false
        )
    }

    static func decodeConfig(
        indexSchedule: String,
        nRoutedExperts: Int,
        nGroup: Int,
        topkGroup: Int
    ) throws -> GLM4MoELiteConfiguration {
        try decodeConfig(
            indexSchedule: indexSchedule,
            nRoutedExperts: nRoutedExperts,
            nGroup: nGroup,
            topkGroup: topkGroup,
            tieWordEmbeddings: false
        )
    }

    static func decodeConfig(
        indexSchedule: String,
        tieWordEmbeddings: Bool
    ) throws -> GLM4MoELiteConfiguration {
        try decodeConfig(
            indexSchedule: indexSchedule,
            nRoutedExperts: 2,
            nGroup: 1,
            topkGroup: 1,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func decodeConfig(
        indexSchedule: String,
        nRoutedExperts: Int,
        nGroup: Int,
        topkGroup: Int,
        tieWordEmbeddings: Bool
    ) throws -> GLM4MoELiteConfiguration {
        try JSONDecoder.json5().decode(
            GLM4MoELiteConfiguration.self,
            from: configJSON(
                indexSchedule: indexSchedule,
                nRoutedExperts: nRoutedExperts,
                nGroup: nGroup,
                topkGroup: topkGroup,
                tieWordEmbeddings: tieWordEmbeddings
            )
        )
    }

    static func writeModelDirectory(indexSchedule: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("GLMMoEDSAArchitectureTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try configJSON(
            indexSchedule: indexSchedule,
            nRoutedExperts: 2,
            nGroup: 1,
            topkGroup: 1,
            tieWordEmbeddings: false
        )
            .write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    static func sanitizedTiedCheckpointWeights() throws -> [String: MLXArray] {
        try Device.withDefaultDevice(.cpu) {
            let config = try decodeConfig(
                indexSchedule: #""index_topk_pattern": "FSFS""#,
                tieWordEmbeddings: true
            )
            let model = GLM4MoELiteModel(config)
            var weights = expertCheckpointWeights()
            weights["lm_head.weight"] = MLXArray.ones([config.vocabularySize, config.hiddenSize])
            weights["model.layers.0.self_attn.kv_b_proj.weight"] = MLXArray.ones([16, 4])
            weights["model.layers.4.self_attn.q_proj.weight"] = MLXArray.ones([1, 1])
            return model.sanitize(weights: weights)
        }
    }

    private static func expertCheckpointWeights() -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for expertIndex in 0 ..< 2 {
            for (name, baseValue) in expertProjectionValues {
                weights["model.layers.0.mlp.experts.\(expertIndex).\(name).weight"] = MLXArray(
                    [Float](repeating: baseValue + Float(expertIndex), count: 4)
                )
                .reshaped([2, 2])
            }
        }
        return weights
    }

    private static func configJSON(
        indexSchedule: String,
        nRoutedExperts: Int,
        nGroup: Int,
        topkGroup: Int,
        tieWordEmbeddings: Bool
    ) -> Data {
        Data("""
        \(baseConfigPrefix)
            "n_group": \(nGroup),
            "n_routed_experts": \(nRoutedExperts),
            "tie_word_embeddings": \(tieWordEmbeddings),
            "topk_group": \(topkGroup),
        \(baseConfigSuffix)
            \(indexSchedule)
        }
        """.utf8)
    }

    private static let expertProjectionValues = [
        ("gate_proj", Float(1)),
        ("down_proj", Float(3)),
        ("up_proj", Float(5))
    ]

    // swiftlint:disable indentation_width
    private static let baseConfigPrefix = """
        {
            "attention_bias": false,
            "first_k_dense_replace": 1,
            "hidden_size": 16,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_topk": 2,
            "intermediate_size": 32,
            "kv_lora_rank": 4,
            "max_position_embeddings": 64,
            "model_type": "glm_moe_dsa",
            "moe_intermediate_size": 8,
            "moe_layer_freq": 1,
        """

    private static let baseConfigSuffix = """
            "n_shared_experts": 1,
            "norm_topk_prob": true,
            "num_attention_heads": 2,
            "num_nextn_predict_layers": 1,
            "num_experts_per_tok": 1,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "q_lora_rank": 6,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "rms_norm_eps": 0.00001,
            "rope_parameters": {"rope_theta": 1000000.0},
            "routed_scaling_factor": 1.0,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "v_head_dim": 4,
            "vocab_size": 64,
        """
    // swiftlint:enable indentation_width
}
