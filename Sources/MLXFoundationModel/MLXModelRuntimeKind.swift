import Foundation

/// Runtime family needed to host a model.
public enum MLXModelRuntimeKind: Codable, CaseIterable, Hashable, Sendable {
    case text
    case vlm

    private static let decodedKinds: [String: Self] = [
        "text": .text,
        "vlm": .vlm
    ]

    private static let encodedKinds = Dictionary(uniqueKeysWithValues: decodedKinds.map { key, value in
        (value, key)
    })

    public init(from decoder: Decoder) throws {
        let value = try String(from: decoder)
        guard let kind = Self.decodedKinds[value] else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported MLX model runtime kind.")
            )
        }
        self = kind
    }

    public func encode(to encoder: Encoder) throws {
        try codingValue.encode(to: encoder)
    }

    public var codingValue: String {
        Self.encodedKinds[self] ?? "text"
    }
}
