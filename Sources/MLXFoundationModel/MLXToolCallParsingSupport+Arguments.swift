import Foundation

extension MLXToolCallParsingSupport {
    static func canonicalArgumentsJSONString(_ value: Any) -> String {
        if let object = value as? JSONObject {
            return canonicalJSONString(object)
        }
        if let string = value as? String,
            let object = parseJSON(string) as? JSONObject {
            return canonicalJSONString(object)
        }
        return "{}"
    }
}
