import Foundation

extension JSONDecoder {
    internal static func json5() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.allowsJSON5 = true
        return decoder
    }
}
