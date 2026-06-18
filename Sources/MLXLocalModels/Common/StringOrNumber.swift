// Copyright © 2024 Apple Inc.

import Foundation

/// Representation of a heterogenous type in a JSON configuration file.
///
/// This can be: a string, a boolean, a numeric value, or an array of numeric values.
/// There are methods to do unwrapping, see e.g. ``asFloat()`` and
/// ``asFloats()`` or callers can switch on the enum.
internal enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case bool(Bool)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? values.decode(Int.self) {
            self = .int(v)
        } else if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else if let v = try? values.decode([Int].self) {
            self = .ints(v)
        } else if let v = try? values.decode([Float].self) {
            self = .floats(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        case .ints(let v): try container.encode(v)
        case .floats(let v): try container.encode(v)
        }
    }

    /// Return the value as an optional array of integers.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    internal func asInts() -> [Int]? {
        switch self {
        case .string(_): nil
        case .bool(_): nil
        case .int(let v): [v]
        case .float(_): nil
        case .ints(let array): array
        case .floats(_): nil
        }
    }

    /// Return the value as an optional integer.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    internal func asInt() -> Int? {
        switch self {
        case .string(_): nil
        case .bool(_): nil
        case .int(let v): v
        case .float(_): nil
        case .ints(let array): array.count == 1 ? array[0] : nil
        case .floats(_): nil
        }
    }

    /// Return the value as an optional array of floats.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    internal func asFloats() -> [Float]? {
        switch self {
        case .string(_): nil
        case .bool(_): nil
        case .int(let v): [Float(v)]
        case .float(let float): [float]
        case .ints(let array): array.map { Float($0) }
        case .floats(let array): array
        }
    }

    /// Return the value as an optional float.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    internal func asFloat() -> Float? {
        switch self {
        case .string(_): nil
        case .bool(_): nil
        case .int(let v): Float(v)
        case .float(let float): float
        case .ints(let array): array.count == 1 ? Float(array[0]) : nil
        case .floats(let array): array.count == 1 ? array[0] : nil
        }
    }

    /// Return the value as an optional boolean.
    internal func asBool() -> Bool? {
        switch self {
        case .bool(let value): value
        case .string, .int, .float, .ints, .floats: nil
        }
    }
}

/// Decodes Hugging Face fields that may be either a single token id or an array of token ids.
internal struct IntOrIntArray: Codable, Equatable, Sendable {
    internal let values: [Int]

    internal init(_ values: [Int]) {
        self.values = values
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            values = [value]
        } else {
            values = try container.decode([Int].self)
        }
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if values.count == 1, let value = values.first {
            try container.encode(value)
        } else {
            try container.encode(values)
        }
    }
}
