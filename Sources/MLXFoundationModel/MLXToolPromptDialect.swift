import Foundation

enum MLXToolPromptDialect {
    case cohereAction
    case deepSeekDSML
    case functionGemma
    case gemma
    case genericJSON
    case glmXML
    case harmony
    case kimiK2
    case longCat
    case minimaxM3
    case minimaxXML
    case mistralToolCall
    case qwenXML

    private static let styleMap: [MLXPromptStyle: Self] = [
        .cohereAction: .cohereAction,
        .deepSeekDSML: .deepSeekDSML,
        .functionGemma: .functionGemma,
        .gemma: .gemma,
        .glmXML: .glmXML,
        .harmony: .harmony,
        .kimiK2: .kimiK2,
        .longCat: .longCat,
        .minimaxM3: .minimaxM3,
        .minimaxXML: .minimaxXML,
        .mistralToolCall: .mistralToolCall,
        .qwenXML: .qwenXML
    ]

    static let miniMaxNamespaceToken = "]<]minimax[>["

    init(style: MLXPromptStyle) {
        self = Self.styleMap[style] ?? .genericJSON
    }

    func renderToolCalls(_ calls: [MLXExtractedToolCall]) -> String {
        if self == .kimiK2 {
            return kimiToolCalls(calls)
        }
        let rendered = calls.enumerated().map { index, call in
            renderToolCall(call, index: index)
        }
        return rendered.joined(separator: "\n")
    }

    func renderToolCall(
        _ call: MLXExtractedToolCall,
        index: Int
    ) -> String {
        if let primary = renderPrimaryToolCall(call, index: index) {
            return primary
        }
        return renderSecondaryToolCall(call, index: index)
    }

    private func renderPrimaryToolCall(
        _ call: MLXExtractedToolCall,
        index: Int
    ) -> String? {
        switch self {
        case .cohereAction:
            return """
            <|START_ACTION|>\
            {"tool_name":"\(call.name)","parameters":\(call.argumentsJSON)}\
            <|END_ACTION|>
            """

        case .deepSeekDSML:
            return deepSeekToolCall(call)

        case .functionGemma:
            return """
            <start_function_call>\
            call:\(call.name)\(MLXGemma4ValueFormatter.legacyArgumentsObject(from: call.argumentsJSON))\
            <end_function_call>
            """

        case .gemma:
            let arguments = MLXGemma4ValueFormatter.argumentsObject(
                from: call.argumentsJSON
            )
            return "<|tool_call>call:\(call.name)\(arguments)<tool_call|>"

        case .genericJSON:
            return MLXToolCallParsingSupport.canonicalJSONString([
                "arguments": parsedArguments(call.argumentsJSON),
                "tool_name": call.name
            ])

        case .glmXML:
            return "<tool_call>\(call.name)\(glmArguments(call.argumentsJSON))</tool_call>"

        case .harmony, .kimiK2, .longCat, .minimaxM3, .minimaxXML, .mistralToolCall, .qwenXML:
            return nil
        }
    }

    private func renderSecondaryToolCall(
        _ call: MLXExtractedToolCall,
        index: Int
    ) -> String {
        switch self {
        case .harmony:
            return harmonyToolCall(call)

        case .kimiK2:
            return kimiToolCall(call, index: index)

        case .longCat:
            return longCatToolCall(call)

        case .minimaxXML:
            return miniMaxXMLToolCall(call)

        case .minimaxM3:
            return miniMaxM3ToolCall(call)

        case .mistralToolCall:
            return "[TOOL_CALLS]\(call.name)[ARGS]\(call.argumentsJSON)"

        case .qwenXML:
            return qwenToolCall(call)

        case .cohereAction, .deepSeekDSML, .functionGemma, .gemma, .genericJSON, .glmXML:
            return renderPrimaryToolCall(call, index: index) ?? ""
        }
    }

    func qwenArguments(_ argumentsJSON: String) -> String {
        argumentPairs(argumentsJSON).map { key, value in
            "<parameter=\(key)>\(argumentText(value))</parameter>"
        }
        .joined()
    }

    private func glmArguments(_ argumentsJSON: String) -> String {
        argumentPairs(argumentsJSON).map { key, value in
            "<arg_key>\(key)</arg_key><arg_value>\(argumentText(value))</arg_value>"
        }
        .joined()
    }

    func minimaxArguments(_ argumentsJSON: String) -> String {
        argumentPairs(argumentsJSON).map { key, value in
            "<parameter name=\"\(key)\">\(argumentText(value))</parameter>"
        }
        .joined()
    }

    func longCatArguments(_ argumentsJSON: String) -> String {
        argumentPairs(argumentsJSON).map { key, value in
            """
            <longcat_arg_key>\(key)</longcat_arg_key>\
            <longcat_arg_value>\(argumentText(value))</longcat_arg_value>
            """
        }
        .joined()
    }

    func argumentPairs(_ argumentsJSON: String) -> [(String, Any)] {
        guard let object = parsedArguments(argumentsJSON) as? [String: Any] else {
            return []
        }
        return object.keys.sorted().map { key in
            (key, object[key] as Any)
        }
    }

    func parsedArguments(_ argumentsJSON: String) -> Any {
        MLXToolCallParsingSupport.parseJSON(argumentsJSON) ?? [:]
    }

    func argumentText(_ value: Any) -> String {
        if let string = value as? String {
            return string
        }
        return MLXToolCallParsingSupport.canonicalJSONString(["value": value])
            .droppingJSONValueEnvelope()
    }
}
