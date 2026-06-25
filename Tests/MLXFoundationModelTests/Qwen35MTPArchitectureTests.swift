import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Qwen3.5 native MTP architecture")
struct Qwen35MTPArchitectureTests {
    @Test("decodes native MTP layer count")
    func decodesNativeMTPLayerCount() throws {
        let config = try Self.decodeConfig(mtpLayers: 2)

        #expect(config.mtpNumHiddenLayers == 2)
        #expect(config.hasNativeMTP)
    }

    @Test("decodes explicit linear and full attention schedule")
    func decodesExplicitLayerSchedule() throws {
        let config = try Self.decodeConfig(
            mtpLayers: 1,
            layerTypes: ["linear_attention", "full_attention"]
        )

        #expect(config.layerTypes == [.linearAttention, .fullAttention])
        #expect(config.layerSchedule.firstLinearIndex == 0)
        #expect(config.layerSchedule.firstFullAttentionIndex == 1)
    }

    @Test("defaults layer schedule from full attention interval")
    func defaultsLayerScheduleFromFullAttentionInterval() throws {
        let config = try Self.decodeConfig(mtpLayers: 0, fullAttentionInterval: 2)

        #expect(config.layerTypes == [.linearAttention, .fullAttention])
    }

    @Test("rejects layer schedule count mismatch")
    func rejectsLayerScheduleCountMismatch() {
        #expect(throws: DecodingError.self) {
            _ = try Self.decodeConfig(
                mtpLayers: 0,
                layerTypes: ["linear_attention"]
            )
        }
    }

    @Test("computes attention and linear-attention layouts")
    func computesAttentionLayouts() throws {
        let config = try Self.decodeConfig(mtpLayers: 0)
        let attention = Qwen35AttentionLayout(config)
        let linear = Qwen35LinearAttentionLayout(config)

        #expect(attention.headDimensions == 8)
        #expect(attention.queryProjectionDimensions == 16)
        #expect(attention.keyValueProjectionDimensions == 16)
        #expect(attention.rotaryDimensions == 2)
        #expect(linear.keyDimensions == 16)
        #expect(linear.valueDimensions == 16)
        #expect(linear.convolutionDimensions == 48)
    }

    @Test("cache types follow decoded layer schedule")
    func cacheTypesFollowDecodedLayerSchedule() throws {
        let config = try Self.decodeConfig(
            mtpLayers: 0,
            layerTypes: ["linear_attention", "full_attention"]
        )
        let model = Qwen35TextModel(config)
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 2)
        #expect(cache[0] is MambaCache)
        #expect(cache[1] is KVCacheSimple)
    }

    @Test("constructs native MTP module when configured")
    func constructsNativeMTPModuleWhenConfigured() throws {
        let model = Qwen35TextModel(try Self.decodeConfig(mtpLayers: 1))

        #expect(model.hasNativeMTP)
        #expect(model.makeMTPCache(parameters: nil).count == 1)
    }

    @Test("sanitizer disables native MTP when projection weights are absent")
    func sanitizerDisablesNativeMTPWhenProjectionWeightsAreAbsent() throws {
        let model = Qwen35TextModel(try Self.decodeConfig(mtpLayers: 1))

        _ = model.sanitize(weights: [:])

        #expect(!model.supportsNativeMTP)
        #expect(model.makeMTPCache(parameters: nil).isEmpty)
    }

    @Test("top-level sanitizer disables config-only native MTP")
    func topLevelSanitizerDisablesConfigOnlyNativeMTP() throws {
        let model = Qwen35Model(try Self.decodeTopLevelConfig(mtpLayers: 1))

        _ = model.sanitize(weights: [:])

        #expect(!model.supportsNativeMTP)
        #expect(model.makeMTPCache(parameters: nil).isEmpty)
    }

    @Test(
        "sanitizer preserves native MTP weights only when configured",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func sanitizerPreservesNativeMTPWeightsOnlyWhenConfigured() throws {
        try Device.withDefaultDevice(.cpu) {
            let mtpModel = Qwen35TextModel(try Self.decodeConfig(mtpLayers: 1))
            let plainModel = Qwen35TextModel(try Self.decodeConfig(mtpLayers: 0))
            let mtpWeight = MLXArray.zeros([16, 16])
            let normalWeight = MLXArray.ones([16, 16])

            let preserved = mtpModel.sanitize(weights: [
                "model.mtp.fc.weight": mtpWeight,
                "model.layers.0.self_attn.q_proj.weight": normalWeight
            ])
            let stripped = plainModel.sanitize(weights: [
                "model.mtp.fc.weight": mtpWeight,
                "model.layers.0.self_attn.q_proj.weight": normalWeight
            ])

            #expect(preserved["mtp.fc.weight"] != nil)
            #expect(preserved["model.mtp.fc.weight"] == nil)
            #expect(preserved["model.layers.0.self_attn.q_proj.weight"] != nil)
            #expect(stripped["mtp.fc.weight"] == nil)
        }
    }

    @Test(
        "top-level sanitizer maps native MTP as language-model sibling",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func topLevelSanitizerMapsNativeMTPAsLanguageModelSibling() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen35Model(try Self.decodeTopLevelConfig(mtpLayers: 1))
            let mtpWeight = MLXArray.zeros([16, 16])
            let sanitized = model.sanitize(weights: [
                "model.mtp.fc.weight": mtpWeight,
                "model.language_model.mtp.norm.weight": MLXArray.ones([16])
            ])

            #expect(sanitized["language_model.mtp.fc.weight"] != nil)
            #expect(sanitized["language_model.mtp.norm.weight"] != nil)
            #expect(sanitized["language_model.model.mtp.fc.weight"] == nil)
            #expect(sanitized["language_model.model.mtp.norm.weight"] == nil)
        }
    }

    @Test(
        "MoE top-level sanitizer maps native MTP as language-model sibling",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func moeTopLevelSanitizerMapsNativeMTPAsLanguageModelSibling() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen35MoEModel(try Self.decodeTopLevelConfig(mtpLayers: 1))
            let mtpWeight = MLXArray.zeros([16, 16])
            let sanitized = model.sanitize(weights: [
                "model.mtp.fc.weight": mtpWeight,
                "model.language_model.mtp.norm.weight": MLXArray.ones([16])
            ])

            #expect(sanitized["language_model.mtp.fc.weight"] != nil)
            #expect(sanitized["language_model.mtp.norm.weight"] != nil)
            #expect(sanitized["language_model.model.mtp.fc.weight"] == nil)
            #expect(sanitized["language_model.model.mtp.norm.weight"] == nil)
        }
    }

    @Test(
        "tiny native MTP module produces finite logits",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func tinyNativeMTPModuleProducesFiniteLogits() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try Self.decodeConfig(mtpLayers: 1)
            let model = Qwen35TextModel(config)
            let hiddenStates = MLXArray.zeros([1, 1, config.hiddenSize])
            let nextTokenIDs = MLXArray([Int32(1)]).reshaped([1, 1])
            let logits = try #require(
                model.mtpForward(
                    hiddenStates: hiddenStates,
                    nextTokenIDs: nextTokenIDs,
                    cache: nil
                )
            )

            eval(logits)

            #expect(logits.shape == [1, 1, config.vocabularySize])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    private static func decodeConfig(
        mtpLayers: Int,
        fullAttentionInterval: Int = 1,
        layerTypes: [String] = []
    ) throws -> Qwen35TextConfiguration {
        try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self,
            from: configJSON(
                mtpLayers: mtpLayers,
                fullAttentionInterval: fullAttentionInterval,
                layerTypes: layerTypes
            )
        )
    }

    private static func decodeTopLevelConfig(mtpLayers: Int) throws -> Qwen35Configuration {
        let textConfigJSON = try #require(String(
            data: configJSON(mtpLayers: mtpLayers),
            encoding: .utf8
        ))
        let data = Data(
            """
            {
                "model_type": "qwen3_5",
                "text_config": \(textConfigJSON)
            }
            """.utf8
        )
        return try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data)
    }

    private static func configJSON(
        mtpLayers: Int,
        fullAttentionInterval: Int = 1,
        layerTypes: [String] = []
    ) -> Data {
        let layerTypesField = Self.layerTypesJSONField(layerTypes)
        return Data(
            """
            {
                "attention_bias": false,
                "full_attention_interval": \(fullAttentionInterval),
                "hidden_size": 16,
                "intermediate_size": 32,
                \(layerTypesField)
                "linear_conv_kernel_dim": 2,
                "linear_key_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_value_head_dim": 8,
                "max_position_embeddings": 64,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": \(mtpLayers),
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 0.000001,
                "rope_theta": 100000.0,
                "tie_word_embeddings": true,
                "vocab_size": 32
            }
            """.utf8
        )
    }

    private static func layerTypesJSONField(_ layerTypes: [String]) -> String {
        guard !layerTypes.isEmpty else {
            return ""
        }
        let values = layerTypes.map { "\"\($0)\"" }.joined(separator: ", ")
        return "\"layer_types\": [\(values)],"
    }
}
