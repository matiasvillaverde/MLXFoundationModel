import Foundation
import MLX
import MLXNN

internal struct KimiK25Configuration: Codable, Equatable, Sendable {
    internal var modelType: String
    internal var textConfig: DeepseekV3Configuration

    internal init(
        modelType: String = "kimi_k25",
        textConfig: DeepseekV3Configuration
    ) {
        self.modelType = modelType
        self.textConfig = textConfig
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "kimi_k25"
        self.textConfig = try container.decode(DeepseekV3Configuration.self, forKey: .textConfig)
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }
}

internal final class KimiK25Model: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "language_model") private var languageModel: DeepseekV3Model

    internal init(_ config: KimiK25Configuration) {
        let languageModel = DeepseekV3Model(config.textConfig)
        self.vocabularySize = languageModel.vocabularySize
        self.kvHeads = languageModel.kvHeads
        self._languageModel.wrappedValue = languageModel
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        languageModel.greedyToken(input, cache: cache, state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var languageWeights: [String: MLXArray] = [:]

        for (key, value) in weights where !Self.isVisionOrProjectorWeight(key) {
            if let stripped = Self.languageWeightKey(from: key) {
                languageWeights[stripped] = value
            } else if Self.isRootLanguageWeight(key) {
                languageWeights[key] = value
            }
        }

        guard !languageWeights.isEmpty else { return [:] }
        return languageModel.sanitize(weights: languageWeights).reduce(into: [:]) { result, entry in
            result["language_model.\(entry.key)"] = entry.value
        }
    }

    private static func languageWeightKey(from key: String) -> String? {
        for prefix in ["language_model.", "model.language_model."] where key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return nil
    }

    private static func isRootLanguageWeight(_ key: String) -> Bool {
        key.hasPrefix("model.") || key.hasPrefix("lm_head.")
    }

    private static func isVisionOrProjectorWeight(_ key: String) -> Bool {
        [
            "vision_tower",
            "vision_model",
            "multi_modal_projector",
            "mm_projector",
            "model.vision_tower",
            "model.vision_model",
            "model.multi_modal_projector",
            "model.mm_projector"
        ].contains { key == $0 || key.hasPrefix("\($0).") }
    }
}

extension KimiK25Model: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        languageModel.loraLinearLayers()
    }
}
