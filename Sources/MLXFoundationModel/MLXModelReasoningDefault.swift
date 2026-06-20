import Foundation

/// Default reasoning behavior inferred from a model's native chat template.
public enum MLXModelReasoningDefault: Codable, CaseIterable, Equatable, Hashable, Sendable {
    case disabled
    case enabled

    private static let decodedValues: [String: Self] = [
        "disabled": .disabled,
        "enabled": .enabled
    ]

    private static let encodedValues = Dictionary(
        uniqueKeysWithValues: decodedValues.map { key, value in
            (value, key)
        }
    )

    public init(from decoder: Decoder) throws {
        let value = try String(from: decoder)
        guard let decoded = Self.decodedValues[value] else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "Unsupported MLX reasoning default."
            ))
        }
        self = decoded
    }

    public func encode(to encoder: Encoder) throws {
        try codingValue.encode(to: encoder)
    }

    public var codingValue: String {
        Self.encodedValues[self] ?? "disabled"
    }
}
