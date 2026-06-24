import Foundation
@testable import MLXLocalModels
import Testing

@Suite("StringOrNumber config values")
struct StringOrNumberTests {
    @Test("decodes supported scalar and array shapes")
    func decodesSupportedShapes() throws {
        #expect(try decode("true") == .bool(true))
        #expect(try decode(#""linear""#) == .string("linear"))
        #expect(try decode("32") == .int(32))
        #expect(try decode("1.25") == .float(1.25))
        #expect(try decode("[1,2,3]") == .ints([1, 2, 3]))
        #expect(try decode("[1.5,2.5]") == .floats([1.5, 2.5]))
    }

    @Test("keeps integer arrays distinct from fractional float arrays")
    func keepsIntegerArraysDistinctFromFractionalFloatArrays() throws {
        let integerArray = try decode("[1,2]")
        let floatArray = try decode("[1.5,2.5]")

        #expect(integerArray == .ints([1, 2]))
        #expect(floatArray == .floats([1.5, 2.5]))
    }

    @Test("converts only numeric values to numeric helpers")
    func convertsNumericValues() {
        #expect(StringOrNumber.int(4).asInt() == 4)
        #expect(StringOrNumber.ints([4]).asInt() == 4)
        #expect(StringOrNumber.ints([4, 8]).asInt() == nil)
        #expect(StringOrNumber.float(2.5).asFloat() == 2.5)
        #expect(StringOrNumber.floats([2.5]).asFloat() == 2.5)
        #expect(StringOrNumber.floats([2.5, 3.5]).asFloat() == nil)
        #expect(StringOrNumber.ints([4, 8]).asFloats() == [4, 8])
        #expect(StringOrNumber.string("4").asInt() == nil)
        #expect(StringOrNumber.bool(true).asFloat() == nil)
    }

    @Test("round trips encoded values")
    func roundTripsEncodedValues() throws {
        let values: [StringOrNumber] = [
            .string("dynamic"),
            .bool(false),
            .int(7),
            .float(7.5),
            .ints([1, 2]),
            .floats([1.25, 2.5])
        ]

        for value in values {
            let encoded = try Self.encoder.encode(value)
            let decoded = try Self.decoder.decode(StringOrNumber.self, from: encoded)

            #expect(decoded == value)
        }
    }

    @Test("decodes and encodes single or multiple token ids")
    func decodesAndEncodesTokenIds() throws {
        let single = try Self.decoder.decode(IntOrIntArray.self, from: Data("7".utf8))
        let many = try Self.decoder.decode(IntOrIntArray.self, from: Data("[7,8]".utf8))

        #expect(single.values == [7])
        #expect(many.values == [7, 8])
        #expect(String(data: try Self.encoder.encode(single), encoding: .utf8) == "7")
        #expect(String(data: try Self.encoder.encode(many), encoding: .utf8) == "[7,8]")
    }

    @Test("rejects unsupported JSON shapes")
    func rejectsUnsupportedShapes() {
        #expect(throws: DecodingError.self) {
            try decode(#"{"value":1}"#)
        }
    }

    private static let decoder = JSONDecoder()
    private static let encoder: JSONEncoder = {
        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = [.sortedKeys]
        return jsonEncoder
    }()

    private func decode(_ json: String) throws -> StringOrNumber {
        try Self.decoder.decode(StringOrNumber.self, from: Data(json.utf8))
    }
}
