import Foundation

/// Structured template policy for model-native reasoning controls.
public struct MLXBridgeReasoningOptions: Codable, Equatable, Hashable, Sendable {
    public enum Effort: Codable, CaseIterable, Hashable, Sendable {
        case light
        case moderate
        case deep

        private static let decodedEfforts: [String: Self] = [
            "light": .light,
            "moderate": .moderate,
            "deep": .deep
        ]

        private static let encodedEfforts = Dictionary(
            uniqueKeysWithValues: decodedEfforts.map { key, value in
                (value, key)
            }
        )

        public init(from decoder: Decoder) throws {
            let value = try String(from: decoder)
            guard let effort = Self.decodedEfforts[value] else {
                throw DecodingError.dataCorrupted(.init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Unsupported MLX reasoning effort."
                ))
            }
            self = effort
        }

        public func encode(to encoder: Encoder) throws {
            try codingValue.encode(to: encoder)
        }

        public var codingValue: String {
            Self.encodedEfforts[self] ?? "moderate"
        }
    }

    public let isEnabled: Bool
    public let effort: Effort?
    public let customEffort: String?

    public init(
        isEnabled: Bool = false,
        effort: Effort? = nil,
        customEffort: String? = nil
    ) {
        self.isEnabled = isEnabled
        self.effort = isEnabled ? effort : nil
        self.customEffort = isEnabled ? Self.trimmedNonEmpty(customEffort) : nil
    }

    public static let disabled = Self()

    public static func enabled(
        effort: Effort? = nil,
        customEffort: String? = nil
    ) -> Self {
        Self(isEnabled: true, effort: effort, customEffort: customEffort)
    }

    private static func trimmedNonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }
}
