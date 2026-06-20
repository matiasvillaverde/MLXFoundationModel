import Foundation

enum MLXOQLevelParser {
    static func detect(
        id: String?,
        config: [String: Any],
        quantizationConfig: [String: Any]
    ) -> String? {
        explicitLevel(in: quantizationConfig)
            ?? explicitLevel(in: config)
            ?? embeddedLevel(in: quantizationConfig)
            ?? embeddedLevel(in: config)
            ?? level(fromIdentifier: id)
    }

    private static func explicitLevel(in metadata: [String: Any]) -> String? {
        let keys = [
            "oq_level",
            "oQ_level",
            "oqLevel",
            "oQLevel",
            "omlx_oq_level",
            "omlxOQLevel"
        ]
        for key in keys {
            guard let value = metadata[key],
                let level = level(fromValue: value)
            else {
                continue
            }
            return level
        }
        return nil
    }

    private static func embeddedLevel(in metadata: [String: Any]) -> String? {
        let keys = [
            "quant_method",
            "quantization_method",
            "method",
            "linear_class",
            "linearClass",
            "quantization_mode",
            "mode",
            "format",
            "dtype"
        ]
        for key in keys {
            guard let value = metadata[key],
                let level = level(fromIdentifier: String(describing: value))
            else {
                continue
            }
            return level
        }
        return nil
    }

    private static func level(fromValue value: Any) -> String? {
        if let intValue = value as? Int {
            return canonicalLevel(String(intValue), suffix: "")
        }
        if let doubleValue = value as? Double {
            return canonicalLevel(levelString(from: doubleValue), suffix: "")
        }
        if let stringValue = value as? String {
            let trimmed = stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            return level(fromIdentifier: trimmed)
                ?? canonicalLevel(trimmed, suffix: "")
        }
        return nil
    }

    private static func level(fromIdentifier value: String?) -> String? {
        guard let value else {
            return nil
        }
        let lowercased = value.lowercased()
        guard let oqRange = lowercased.range(of: "oq")
        else {
            return nil
        }
        var current = value.index(value.startIndex, offsetBy: lowercased.distance(
            from: lowercased.startIndex,
            to: oqRange.upperBound
        ))
        current = skipSeparators(in: value, startingAt: current)
        let levelStart = current
        current = skipLevelCharacters(in: value, startingAt: current)
        let level = value[levelStart..<current]
        let suffixStart = current
        current = skipSuffixCharacters(in: value, startingAt: current)
        return canonicalLevel(String(level), suffix: value[suffixStart..<current].lowercased())
    }

    private static func skipSeparators(
        in value: String,
        startingAt index: String.Index
    ) -> String.Index {
        var current = index
        while current < value.endIndex,
            isSeparator(value[current]) {
            current = value.index(after: current)
        }
        return current
    }

    private static func skipLevelCharacters(
        in value: String,
        startingAt index: String.Index
    ) -> String.Index {
        var current = index
        while current < value.endIndex,
            isLevelCharacter(value[current]) {
            current = value.index(after: current)
        }
        return current
    }

    private static func skipSuffixCharacters(
        in value: String,
        startingAt index: String.Index
    ) -> String.Index {
        var current = index
        while current < value.endIndex,
            isSuffixCharacter(value[current]) {
            current = value.index(after: current)
        }
        return current
    }

    private static func canonicalLevel(_ value: String, suffix: String) -> String? {
        let level = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard supportedLevels.contains(level) else {
            return nil
        }
        return "oQ\(level)\(suffix)"
    }

    private static let supportedLevels: Set<String> = [
        "2",
        "2.5",
        "2.7",
        "3",
        "3.5",
        "4",
        "5",
        "6",
        "8"
    ]

    private static func isSeparator(_ character: Character) -> Bool {
        character == "-" || character == "_" || character == " "
    }

    private static func isLevelCharacter(_ character: Character) -> Bool {
        character.isNumber || character == "."
    }

    private static func isSuffixCharacter(_ character: Character) -> Bool {
        character.isLetter
    }

    private static func levelString(from value: Double) -> String {
        value.rounded() == value ? String(Int(value)) : String(value)
    }
}
