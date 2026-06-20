import Foundation

/// Supported oMLX Universal Dynamic Quantization level.
public struct MLXOQLevel: Codable, Equatable, Hashable, Sendable {
    public let value: String

    public init?(_ value: Double) {
        self.init(String(format: "%g", value))
    }

    public init?(_ value: String) {
        guard let level = Self.canonicalLevel(from: value) else {
            return nil
        }
        self.value = level
    }

    public var label: String {
        "oQ\(value)"
    }

    public var baseBits: Int {
        switch value {
        case "2", "2.5", "2.7":
            2

        case "3", "3.5":
            3

        case "4":
            4

        case "5":
            5

        case "6":
            6

        case "8":
            8

        default:
            4
        }
    }

    var routedExpertDownProjectionBoost: Int? {
        switch value {
        case "2.5", "3.5":
            1

        case "2.7":
            2

        default:
            nil
        }
    }

    private static func canonicalLevel(from value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let extracted = numericSuffix(from: trimmed) ?? trimmed
        guard supportedLevels.contains(extracted) else {
            return nil
        }
        return extracted
    }

    private static func numericSuffix(from value: String) -> String? {
        let lowercased = value.lowercased()
        guard let oqRange = lowercased.range(of: "oq") else {
            return nil
        }
        var current = oqRange.upperBound
        while current < lowercased.endIndex,
            isSeparator(lowercased[current]) {
            current = lowercased.index(after: current)
        }
        let start = current
        while current < lowercased.endIndex,
            isLevelCharacter(lowercased[current]) {
            current = lowercased.index(after: current)
        }
        guard start < current else {
            return nil
        }
        return String(lowercased[start..<current])
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
}
