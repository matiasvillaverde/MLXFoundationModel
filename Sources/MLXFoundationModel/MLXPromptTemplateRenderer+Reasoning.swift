import Foundation

extension MLXPromptTemplateRenderer {
    static func generationStartsInReasoning(
        reasoningEnabled: Bool,
        style: MLXPromptStyle
    ) -> Bool {
        generationStartsInReasoning(
            reasoningOptions: MLXBridgeReasoningOptions(isEnabled: reasoningEnabled),
            style: style
        )
    }

    static func generationStartsInReasoning(
        reasoningOptions: MLXBridgeReasoningOptions,
        style: MLXPromptStyle
    ) -> Bool {
        switch style {
        case .minimaxXML:
            return true

        case .chatML,
            .deepSeekDSML,
            .functionGemma,
            .gemma,
            .glmXML,
            .longCat,
            .minimaxM3,
            .plain,
            .qwenXML:
            return reasoningOptions.isEnabled

        case .cohereAction:
            return reasoningOptions.isEnabled

        case .apertus,
            .harmony,
            .kimiK2,
            .mistralToolCall:
            return false
        }
    }

    static func reasoningOpenSuffix(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        guard request.effectiveReasoningOptions.isEnabled else {
            return ""
        }
        return reasoningOpenMarker(for: style) ?? ""
    }

    private static func reasoningOpenMarker(for style: MLXPromptStyle) -> String? {
        switch style {
        case .gemma:
            return "<|channel>thought\n"

        case .apertus, .harmony:
            return nil

        case .minimaxM3:
            return "<mm:think>"

        case .minimaxXML:
            return nil

        case .longCat:
            return "<longcat_think>\n"

        case .cohereAction:
            return "<|START_THINKING|>"

        case .functionGemma, .glmXML, .kimiK2:
            return "<think>\n"

        case .mistralToolCall, .plain, .chatML, .deepSeekDSML, .qwenXML:
            return "<think>\n"
        }
    }
}
