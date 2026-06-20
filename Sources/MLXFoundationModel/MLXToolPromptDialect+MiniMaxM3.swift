import Foundation

extension MLXToolPromptDialect {
    func miniMaxM3ToolCall(_ call: MLXExtractedToolCall) -> String {
        """
        \(Self.miniMaxNamespaceToken)<tool_call>
        \(Self.miniMaxNamespaceToken)<invoke name="\(call.name)">\
        \(miniMaxM3Arguments(call.argumentsJSON))\
        \(Self.miniMaxNamespaceToken)</invoke>
        \(Self.miniMaxNamespaceToken)</tool_call>
        """
    }

    private func miniMaxM3Arguments(_ argumentsJSON: String) -> String {
        argumentPairs(argumentsJSON).map { key, value in
            miniMaxM3Element(key: key, value: value)
        }
        .joined()
    }

    private func miniMaxM3Value(_ value: Any) -> String {
        if let object = value as? [String: Any] {
            return miniMaxM3Object(object)
        }
        if let array = value as? [Any] {
            return miniMaxM3Array(array)
        }
        return argumentText(value)
    }

    private func miniMaxM3Object(_ object: [String: Any]) -> String {
        object.keys
            .sorted()
            .map { key in
                miniMaxM3Element(key: key, value: object[key] as Any)
            }
        .joined()
    }

    private func miniMaxM3Array(_ array: [Any]) -> String {
        array.map { item in
            miniMaxM3Element(key: "item", value: item)
        }
        .joined()
    }

    private func miniMaxM3Element(key: String, value: Any) -> String {
        """
        \(Self.miniMaxNamespaceToken)<\(key)>\
        \(miniMaxM3Value(value))\
        \(Self.miniMaxNamespaceToken)</\(key)>
        """
    }
}
