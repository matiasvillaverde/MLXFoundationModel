import Foundation

/// Model-level traits used by the oQ predicate chain.
public struct MLXOQModelQuantizationTraits: Codable, Equatable, Hashable, Sendable {
    public let numLayers: Int
    public let numExperts: Int
    public let hiddenSize: Int

    public init(
        numLayers: Int = 32,
        numExperts: Int = 0,
        hiddenSize: Int = 0
    ) {
        self.numLayers = max(1, numLayers)
        self.numExperts = max(0, numExperts)
        self.hiddenSize = max(0, hiddenSize)
    }

    public static func make(config: [String: Any]) -> Self {
        let configs = nestedConfigs(from: config)
        return Self(
            numLayers: int(configs, keys: ["num_hidden_layers"]) ?? 32,
            numExperts: int(
                configs,
                keys: ["num_local_experts", "num_experts", "n_routed_experts"]
            ) ?? 0,
            hiddenSize: int(configs, keys: ["hidden_size"]) ?? 0
        )
    }

    public var isMixtureOfExperts: Bool {
        numExperts > 0
    }

    private static func nestedConfigs(from config: [String: Any]) -> [[String: Any]] {
        let nestedKeys = [
            "text_config",
            "language_model",
            "language_model_config",
            "llm_config"
        ]
        let nested = nestedKeys.compactMap { config[$0] as? [String: Any] }
        return [config] + nested
    }

    private static func int(
        _ configs: [[String: Any]],
        keys: [String]
    ) -> Int? {
        for config in configs {
            for key in keys {
                guard let value = int(config[key]) else {
                    continue
                }
                return value
            }
        }
        return nil
    }

    private static func int(_ value: Any?) -> Int? {
        if let value = value as? Int, value > 0 {
            return value
        }
        if let value = value as? Double, value > 0 {
            return Int(value)
        }
        if let value = value as? String,
            let parsed = Int(value),
            parsed > 0 {
            return parsed
        }
        return nil
    }
}
