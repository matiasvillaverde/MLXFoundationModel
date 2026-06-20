import Foundation

extension MLXModelProfileFactory {
    static func loadJSON(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.coderInvalidValue)
        }
        return object
    }

    static func loadOptionalJSON(at url: URL) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return [:]
        }
        return try loadJSON(at: url)
    }

    static func loadOptionalText(at url: URL) throws -> String? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    static func contextLength(_ config: [String: Any]) -> Int? {
        int(config, keys: [
            "max_position_embeddings",
            "model_max_length",
            "n_positions",
            "seq_length",
            "max_sequence_length"
        ])
    }

    static func chatTemplate(
        config: [String: Any],
        tokenizerConfig: [String: Any],
        standaloneTemplate: String? = nil
    ) -> String? {
        if let standaloneTemplate, !standaloneTemplate.isEmpty {
            return standaloneTemplate
        }
        if let template = string(tokenizerConfig, keys: ["chat_template"]) {
            return template
        }
        if let template = string(config, keys: ["chat_template"]) {
            return template
        }
        return serializedTemplate(tokenizerConfig["chat_template"] ?? config["chat_template"])
    }

    static func quantizationBits(_ config: [String: Any]) -> Int? {
        if let bits = int(config, keys: ["bits"]) {
            return bits
        }
        for key in ["quantization", "quantization_config"] {
            if let nested = config[key] as? [String: Any],
                let bits = int(nested, keys: ["bits", "nbits"]) {
                return bits
            }
        }
        return nil
    }

    static func string(_ config: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = config[key] as? String {
                return value
            }
        }
        return nil
    }

    static func stringArray(_ config: [String: Any], keys: [String]) -> [String] {
        for key in keys {
            if let values = config[key] as? [String] {
                return values
            }
            if let value = config[key] as? String {
                return [value]
            }
        }
        return []
    }

    static func int(_ config: [String: Any], keys: [String]) -> Int? {
        for key in keys {
            if let value = config[key] as? Int, value > 0 {
                return value
            }
            if let value = config[key] as? Double, value > 0 {
                return Int(value)
            }
            if let value = config[key] as? String, let parsed = Int(value), parsed > 0 {
                return parsed
            }
        }
        return nil
    }

    private static func serializedTemplate(_ value: Any?) -> String? {
        guard let value else {
            return nil
        }
        guard JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value),
            let text = String(data: data, encoding: .utf8)
        else {
            return nil
        }
        return text
    }
}
