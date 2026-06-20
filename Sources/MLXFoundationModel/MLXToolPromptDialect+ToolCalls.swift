import Foundation

extension MLXToolPromptDialect {
    func harmonyToolCall(_ call: MLXExtractedToolCall) -> String {
        """
        <|start|>assistant to=functions.\(call.name)<|channel|>commentary\
        <|message|>\(call.argumentsJSON)<|call|>
        """
    }

    func kimiToolCall(
        _ call: MLXExtractedToolCall,
        index: Int
    ) -> String {
        kimiToolCalls([(index, call)])
    }

    func kimiToolCalls(_ calls: [MLXExtractedToolCall]) -> String {
        kimiToolCalls(Array(calls.enumerated()))
    }

    private func kimiToolCalls(_ calls: [(offset: Int, element: MLXExtractedToolCall)]) -> String {
        guard !calls.isEmpty else {
            return ""
        }
        let body = calls.map { index, call in
            """
            <|tool_call_begin|>functions.\(call.name):\(index)\
            <|tool_call_argument_begin|>\(call.argumentsJSON)<|tool_call_end|>
            """
        }
        .joined()
        return "<|tool_calls_section_begin|>\(body)<|tool_calls_section_end|>"
    }

    func longCatToolCall(_ call: MLXExtractedToolCall) -> String {
        let payload: [String: Any] = [
            "arguments": parsedArguments(call.argumentsJSON),
            "name": call.name
        ]
        return """
        <longcat_tool_call>\
        \(MLXToolCallParsingSupport.canonicalJSONString(payload))\
        </longcat_tool_call>
        """
    }

    func miniMaxXMLToolCall(_ call: MLXExtractedToolCall) -> String {
        """
        <minimax:tool_call><invoke name="\(call.name)">\
        \(minimaxArguments(call.argumentsJSON))</invoke></minimax:tool_call>
        """
    }

    func qwenToolCall(_ call: MLXExtractedToolCall) -> String {
        """
        <tool_call><function=\(call.name)>\
        \(qwenArguments(call.argumentsJSON))\
        </function></tool_call>
        """
    }
}
