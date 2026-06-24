import Foundation

/// JSON scalar used by model configuration fields that are not stable across
/// model families.
internal enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case bool(Bool)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])

    internal init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()

        if let decoded = try? value.decode(Bool.self) {
            self = .bool(decoded)
        } else if let decoded = try? value.decode(Int.self) {
            self = .int(decoded)
        } else if let decoded = try? value.decode(Float.self) {
            self = .float(decoded)
        } else if let decoded = try? value.decode([Int].self) {
            self = .ints(decoded)
        } else if let decoded = try? value.decode([Float].self) {
            self = .floats(decoded)
        } else {
            self = .string(try value.decode(String.self))
        }
    }

    internal func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()

        switch self {
        case .string(let string):
            try value.encode(string)
        case .bool(let bool):
            try value.encode(bool)
        case .int(let int):
            try value.encode(int)
        case .float(let float):
            try value.encode(float)
        case .ints(let ints):
            try value.encode(ints)
        case .floats(let floats):
            try value.encode(floats)
        }
    }

    internal func asInts() -> [Int]? {
        switch self {
        case .int(let int):
            [int]
        case .ints(let ints):
            ints
        case .string, .bool, .float, .floats:
            nil
        }
    }

    internal func asInt() -> Int? {
        switch self {
        case .int(let int):
            int
        case .ints(let ints) where ints.count == 1:
            ints[0]
        case .string, .bool, .float, .ints, .floats:
            nil
        }
    }

    internal func asFloats() -> [Float]? {
        switch self {
        case .int(let int):
            [Float(int)]
        case .float(let float):
            [float]
        case .ints(let ints):
            ints.map(Float.init)
        case .floats(let floats):
            floats
        case .string, .bool:
            nil
        }
    }

    internal func asFloat() -> Float? {
        switch self {
        case .int(let int):
            Float(int)
        case .float(let float):
            float
        case .ints(let ints) where ints.count == 1:
            Float(ints[0])
        case .floats(let floats) where floats.count == 1:
            floats[0]
        case .string, .bool, .ints, .floats:
            nil
        }
    }

    internal func asBool() -> Bool? {
        if case .bool(let bool) = self {
            return bool
        }
        return nil
    }
}

/// Hugging Face token id field that may be represented as one id or many ids.
internal struct IntOrIntArray: Codable, Equatable, Sendable {
    internal let values: [Int]

    internal init(_ values: [Int]) {
        self.values = values
    }

    internal init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        values = if let id = try? value.decode(Int.self) {
            [id]
        } else {
            try value.decode([Int].self)
        }
    }

    internal func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        if let onlyValue = values.onlyElement {
            try value.encode(onlyValue)
        } else {
            try value.encode(values)
        }
    }
}

private extension Collection {
    var onlyElement: Element? {
        count == 1 ? first : nil
    }
}
