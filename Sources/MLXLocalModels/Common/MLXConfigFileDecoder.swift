import Foundation
import Hub

package enum MLXConfigFileDecoder {
    package static func loadDictionary(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder.json5().decode(Config.self, from: data)
        guard let dictionary = dictionary(from: config) else {
            throw CocoaError(.coderInvalidValue)
        }
        return dictionary
    }

    private static func dictionary(from config: Config) -> [String: Any]? {
        guard let entries = config.dictionary() else {
            return nil
        }
        return Dictionary(uniqueKeysWithValues: entries.map { key, value in
            (key.string, jsonValue(from: value))
        })
    }

    private static func jsonValue(from config: Config) -> Any {
        if let dictionary = dictionary(from: config) {
            return dictionary
        }
        if let array = config.array() {
            return array.map(jsonValue(from:))
        }
        if let integer = config.integer() {
            return integer
        }
        if let floating = config.floating() {
            return Double(floating)
        }
        if let boolean = config.boolean() {
            return boolean
        }
        if let string = config.string() {
            return string
        }
        return NSNull()
    }
}
