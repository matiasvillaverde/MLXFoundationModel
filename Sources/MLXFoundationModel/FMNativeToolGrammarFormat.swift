import Foundation

enum FMNativeToolGrammarFormat: String {
    enum RuleKind {
        case functionGemma
        case gemma
        case jsonEnvelope
        case xmlParameters
    }

    case cohereAction = "cohere_action"
    case deepSeekDSML = "deep_seek_dsml"
    case functionGemma = "function_gemma"
    case gemma = "gemma"
    case glmXML = "glm_xml"
    case harmony = "harmony"
    case kimiK2 = "kimi_k2"
    case longCat = "long_cat"
    case minimaxM3 = "minimax_m3"
    case minimaxXML = "minimax_xml"
    case mistralToolCall = "mistral_tool_call"
    case qwenXML = "qwen_xml"

    init?(promptStyle: MLXPromptStyle) {
        guard let format = Self.formatsByPromptStyle[promptStyle] else {
            return nil
        }
        self = format
    }

    var ruleKind: RuleKind {
        if self == .functionGemma {
            return .functionGemma
        }
        if self == .gemma {
            return .gemma
        }
        if Self.jsonEnvelopeFormats.contains(self) {
            return .jsonEnvelope
        }
        return .xmlParameters
    }

    var suffix: String {
        switch self {
        case .deepSeekDSML:
            Self.literal("</｜DSML｜invoke>\n</｜DSML｜tool_calls>")

        case .functionGemma:
            Self.literal("<end_function_call>")

        case .gemma:
            Self.literal("<tool_call|>")

        case .glmXML:
            Self.literal("</tool_call>")

        case .minimaxM3:
            Self.literal("]<]minimax[>[</invoke>")
                + " " + Self.literal("]<]minimax[>[</tool_call>")

        case .minimaxXML:
            Self.literal("</invoke></minimax:tool_call>")

        case .qwenXML:
            Self.literal("</function></tool_call>")

        default:
            Self.literal("")
        }
    }

    func prefix(toolName: String) -> String {
        switch self {
        case .deepSeekDSML:
            Self.literal("\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"\(toolName)\">\n")

        case .functionGemma:
            Self.literal("<start_function_call>call:\(toolName)")

        case .gemma:
            Self.literal("<|tool_call>call:\(toolName)")

        case .glmXML:
            Self.literal("<tool_call>\(toolName)")

        case .minimaxM3:
            minimaxM3Prefix(toolName: toolName)

        case .minimaxXML:
            Self.literal("<minimax:tool_call><invoke name=\"\(toolName)\">")

        case .qwenXML:
            Self.literal("<tool_call><function=\(toolName)>")

        default:
            Self.literal("")
        }
    }

    var parameterFallbackRule: String {
        switch self {
        case .deepSeekDSML:
            "native_xml_text"

        case .minimaxM3:
            "native_bracket_text"

        case .functionGemma, .gemma:
            "native_scalar_text"

        default:
            "native_xml_text"
        }
    }

    func parameter(
        parameterName: String,
        valueRule: String,
        stringEncoded: Bool = true
    ) -> String {
        switch self {
        case .deepSeekDSML:
            deepSeekParameter(
                parameterName: parameterName,
                valueRule: valueRule,
                stringEncoded: stringEncoded
            )

        case .glmXML:
            xmlParameter(
                keyStart: "<arg_key>",
                valueStart: "</arg_key><arg_value>",
                parameterName: parameterName,
                valueRule: valueRule
            )

        case .minimaxM3:
            minimaxM3Parameter(parameterName: parameterName, valueRule: valueRule)

        case .minimaxXML:
            minimaxXMLParameter(parameterName: parameterName, valueRule: valueRule)

        case .qwenXML:
            qwenParameter(parameterName: parameterName, valueRule: valueRule)

        default:
            Self.literal("")
        }
    }

    private static let formatsByPromptStyle: [MLXPromptStyle: Self] = [
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

    private static let jsonEnvelopeFormats: Set<Self> = [
        .cohereAction,
        .harmony,
        .kimiK2,
        .longCat,
        .mistralToolCall
    ]

    private func minimaxM3Prefix(toolName: String) -> String {
        Self.literal("]<]minimax[>[<tool_call>")
            + " " + Self.literal("]<]minimax[>[<invoke name=\"\(toolName)\">")
    }

    private func deepSeekParameter(
        parameterName: String,
        valueRule: String,
        stringEncoded: Bool
    ) -> String {
        Self.literal("<｜DSML｜parameter name=\"")
            + " \(parameterName) "
            + Self.literal("\" string=\"\(stringEncoded)\">")
            + " \(valueRule) "
            + Self.literal("</｜DSML｜parameter>")
    }

    private func xmlParameter(
        keyStart: String,
        valueStart: String,
        parameterName: String,
        valueRule: String
    ) -> String {
        Self.literal(keyStart)
            + " \(parameterName) "
            + Self.literal(valueStart)
            + " \(valueRule) "
            + Self.literal("</arg_value>")
    }

    private func minimaxM3Parameter(parameterName: String, valueRule: String) -> String {
        Self.literal("]<]minimax[>[<")
            + " \(parameterName) "
            + Self.literal(">")
            + " \(valueRule) "
            + Self.literal("]<]minimax[>[</")
            + " \(parameterName) "
            + Self.literal(">")
    }

    private func minimaxXMLParameter(parameterName: String, valueRule: String) -> String {
        namedXMLParameter(
            prefix: "<parameter name=\"",
            nameTerminator: "\">",
            suffix: "</parameter>",
            parameterName: parameterName,
            valueRule: valueRule
        )
    }

    private func qwenParameter(parameterName: String, valueRule: String) -> String {
        namedXMLParameter(
            prefix: "<parameter=",
            nameTerminator: ">",
            suffix: "</parameter>",
            parameterName: parameterName,
            valueRule: valueRule
        )
    }

    private func namedXMLParameter(
        prefix: String,
        nameTerminator: String,
        suffix: String,
        parameterName: String,
        valueRule: String
    ) -> String {
        Self.literal(prefix)
            + " \(parameterName) "
            + Self.literal(nameTerminator)
            + " \(valueRule) "
            + Self.literal(suffix)
    }

    static func literal(_ value: String) -> String {
        #""\#(escapedEBNFLiteral(value))""#
    }

    private static func escapedEBNFLiteral(_ value: String) -> String {
        value.reduce(into: "") { result, character in
            switch character {
            case "\"":
                result += #"\""#

            case "\\":
                result += #"\\"#

            case "\n":
                result += #"\n"#

            case "\r":
                result += #"\r"#

            case "\t":
                result += #"\t"#

            default:
                result.append(character)
            }
        }
    }
}
