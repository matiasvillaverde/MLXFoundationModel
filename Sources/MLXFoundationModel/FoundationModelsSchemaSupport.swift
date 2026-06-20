#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsSchemaSupport {
    static func jsonSchemaString(from schema: GenerationSchema) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard
            let data = try? encoder.encode(schema),
            let string = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return string
    }

    static func jsonValue(from string: String) -> Any {
        guard
            let data = string.data(using: .utf8),
            let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else {
            return string
        }
        return value
    }

    static func stringChoices(from schema: GenerationSchema) -> [String] {
        let value = jsonValue(from: jsonSchemaString(from: schema))
        return stringChoices(from: value)
    }

    static func jsonString(from value: Any) -> String {
        guard
            JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let string = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return string
    }

    private static func stringChoices(from value: Any) -> [String] {
        if let choices = value as? [String] {
            return normalizedChoices(choices)
        }
        guard let object = value as? [String: Any] else {
            return []
        }
        let enumChoices = choicesArray(from: object["enum"])
        if !enumChoices.isEmpty {
            return enumChoices
        }
        let typeChoices = choicesArray(from: object["anyOf"])
        if !typeChoices.isEmpty {
            return typeChoices
        }
        if let choice = object["const"] as? String {
            return normalizedChoices([choice])
        }
        for key in ["anyOf", "oneOf"] {
            let choices = unionChoices(from: object[key])
            if !choices.isEmpty {
                return choices
            }
        }
        return []
    }

    private static func unionChoices(from value: Any?) -> [String] {
        guard let values = value as? [Any] else {
            return []
        }
        var choices: [String] = []
        for value in values {
            let nested = stringChoices(from: value)
            guard !nested.isEmpty else {
                return []
            }
            choices.append(contentsOf: nested)
        }
        return normalizedChoices(choices)
    }

    private static func choicesArray(from value: Any?) -> [String] {
        guard let choices = value as? [String] else {
            return []
        }
        return normalizedChoices(choices)
    }

    private static func normalizedChoices(_ choices: [String]) -> [String] {
        var seen: Set<String> = []
        return choices.filter { choice in
            !choice.isEmpty && seen.insert(choice).inserted
        }
    }
}
#endif
