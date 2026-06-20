import Foundation

enum MLXGemma4ValueFormatter {
    typealias JSONObject = [String: Any]

    static func toolDeclaration(_ tool: MLXBridgeToolDefinition) -> String {
        let schema = MLXToolCallParsingSupport.parseJSON(tool.parametersJSONSchema)
            as? JSONObject ?? defaultParameterSchema()
        let description = tool.description.isEmpty
            ? ""
            : "description:\(argument(tool.description)),"
        return """
        <|tool>declaration:\(tool.name){\(description)parameters:\(schemaValue(schema))}<tool|>
        """
    }

    static func argumentsObject(from argumentsJSON: String) -> String {
        guard let object = MLXToolCallParsingSupport.parseJSON(argumentsJSON) as? JSONObject else {
            return "{}"
        }
        return objectLiteral(object, transform: argument)
    }

    static func legacyArgumentsObject(from argumentsJSON: String) -> String {
        guard let object = MLXToolCallParsingSupport.parseJSON(argumentsJSON) as? JSONObject else {
            return "{}"
        }
        return objectLiteral(object, transform: legacyArgument)
    }

    static func toolResponse(name: String, content: String) -> String {
        let parsed = MLXToolCallParsingSupport.parseJSON(content) ?? content
        return "<|tool_response>response:\(name)\(responseObject(parsed))<tool_response|>"
    }

    private static func responseObject(_ response: Any) -> String {
        if let object = response as? JSONObject {
            return objectLiteral(object, transform: argument)
        }
        return "{value:\(argument(response))}"
    }

    private static func schemaValue(_ value: Any) -> String {
        if let object = value as? JSONObject {
            return objectLiteral(object) { key, value in
                schemaArgument(value, key: key)
            }
        }
        if let values = value as? [Any] {
            return "[\(values.map(schemaValue).joined(separator: ","))]"
        }
        return argument(value)
    }

    private static func schemaArgument(_ value: Any, key: String) -> String {
        if key == "type", let string = value as? String {
            return argument(string.uppercased())
        }
        return schemaValue(value)
    }

    private static func objectLiteral(
        _ object: JSONObject,
        transform: (Any) -> String
    ) -> String {
        objectLiteral(object) { _, value in
            transform(value)
        }
    }

    private static func objectLiteral(
        _ object: JSONObject,
        transform: (String, Any) -> String
    ) -> String {
        let body = object.keys
            .sorted()
            .map { key in
                "\(key):\(transform(key, object[key] ?? NSNull()))"
            }
            .joined(separator: ",")
        return "{\(body)}"
    }

    private static func argument(_ value: Any) -> String {
        if let string = value as? String {
            return #"<|"|>\#(string)<|"|>"#
        }
        if let values = value as? [Any] {
            return "[\(values.map(argument).joined(separator: ","))]"
        }
        if let object = value as? JSONObject {
            return objectLiteral(object, transform: argument)
        }
        if let bool = value as? Bool {
            return bool ? "true" : "false"
        }
        if value is NSNull {
            return "null"
        }
        return String(describing: value)
    }

    private static func legacyArgument(_ value: Any) -> String {
        if let string = value as? String {
            return string
        }
        return argument(value)
    }

    private static func defaultParameterSchema() -> JSONObject {
        [
            "properties": [:] as JSONObject,
            "type": "object"
        ]
    }
}
