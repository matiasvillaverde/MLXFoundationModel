import Foundation

extension FMNativeToolGrammarFormat {
    func simpleJSONEnvelope(toolName: String, payloadRule: String) -> String {
        switch self {
        case .cohereAction:
            cohereEnvelope(toolName: toolName, payloadRule: payloadRule)

        case .harmony:
            harmonyEnvelope(toolName: toolName, payloadRule: payloadRule)

        case .kimiK2:
            kimiEnvelope(toolName: toolName, payloadRule: payloadRule)

        case .longCat:
            longCatEnvelope(toolName: toolName, payloadRule: payloadRule)

        case .mistralToolCall:
            Self.literal("[TOOL_CALLS]\(toolName)[ARGS]") + " \(payloadRule)"

        default:
            Self.literal("")
        }
    }

    private func cohereEnvelope(toolName: String, payloadRule: String) -> String {
        Self.literal("<|START_ACTION|>{\"tool_name\":\"\(toolName)\",\"parameters\":")
            + " \(payloadRule) "
            + Self.literal("}<|END_ACTION|>")
    }

    private func harmonyEnvelope(toolName: String, payloadRule: String) -> String {
        Self.literal("<|start|>assistant to=functions.\(toolName)<|channel|>commentary<|message|>")
            + " \(payloadRule) "
            + Self.literal("<|call|>")
    }

    private func kimiEnvelope(toolName: String, payloadRule: String) -> String {
        Self.literal(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.\(toolName):0"
                + "<|tool_call_argument_begin|>"
        )
            + " \(payloadRule) "
            + Self.literal("<|tool_call_end|><|tool_calls_section_end|>")
    }

    private func longCatEnvelope(toolName: String, payloadRule: String) -> String {
        Self.literal("<longcat_tool_call>{\"arguments\":")
            + " \(payloadRule) "
            + Self.literal(",\"name\":\"\(toolName)\"}</longcat_tool_call>")
    }
}
