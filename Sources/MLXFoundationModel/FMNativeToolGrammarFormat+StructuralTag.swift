import Foundation

extension FMNativeToolGrammarFormat {
    var structuralTagSchemaStyle: String? {
        switch self {
        case .deepSeekDSML:
            "deepseek_xml"

        case .glmXML:
            "glm_xml"

        case .minimaxXML:
            "minimax_xml"

        case .qwenXML:
            "qwen_xml"

        default:
            nil
        }
    }

    func structuralTagBegin(toolName: String) -> String {
        switch self {
        case .deepSeekDSML:
            "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"\(toolName)\">\n"

        case .glmXML:
            "<tool_call>\(toolName)"

        case .minimaxXML:
            "<minimax:tool_call><invoke name=\"\(toolName)\">"

        case .qwenXML:
            "<tool_call><function=\(toolName)>"

        default:
            ""
        }
    }

    var structuralTagEnd: String {
        switch self {
        case .deepSeekDSML:
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"

        case .glmXML:
            "</tool_call>"

        case .minimaxXML:
            "</invoke></minimax:tool_call>"

        case .qwenXML:
            "</function></tool_call>"

        default:
            ""
        }
    }
}
