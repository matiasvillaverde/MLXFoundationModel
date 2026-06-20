import Foundation

extension MLXToolPromptDialect {
    static let deepSeekDSMLToken = "｜DSML｜"

    func deepSeekToolCall(_ call: MLXExtractedToolCall) -> String {
        """
        <\(Self.deepSeekDSMLToken)tool_calls>
        <\(Self.deepSeekDSMLToken)invoke name="\(call.name)">
        \(deepSeekArguments(call.argumentsJSON))
        </\(Self.deepSeekDSMLToken)invoke>
        </\(Self.deepSeekDSMLToken)tool_calls>
        """
    }

    func deepSeekArguments(_ argumentsJSON: String) -> String {
        argumentPairs(argumentsJSON).map { key, value in
            """
            <\(Self.deepSeekDSMLToken)parameter name="\(key)" string="\(value is String)">\
            \(deepSeekArgumentText(value))\
            </\(Self.deepSeekDSMLToken)parameter>
            """
        }
        .joined(separator: "\n")
    }

    private func deepSeekArgumentText(_ value: Any) -> String {
        if let string = value as? String {
            return string
        }
        return MLXToolCallParsingSupport.canonicalJSONString(["value": value])
            .droppingJSONValueEnvelope()
    }
}
