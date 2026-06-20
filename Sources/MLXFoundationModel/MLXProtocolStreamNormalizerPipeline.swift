import Foundation

struct MLXProtocolStreamNormalizerPipeline {
    private enum Stage {
        case cohere(MLXCohereCommandStreamFilter)
        case harmony(MLXHarmonyStreamFilter)
        case gemma4(MLXGemma4StreamFilter)
        case longCat(MLXLongCatStreamFilter)
        case miniMaxM3(MLXMiniMaxM3StreamFilter)

        mutating func feed(_ text: String) -> String {
            switch self {
            case .cohere(var filter):
                let output = filter.feed(text)
                self = .cohere(filter)
                return output

            case .harmony(var filter):
                let output = filter.feed(text)
                self = .harmony(filter)
                return output

            case .gemma4(var filter):
                let output = filter.feed(text)
                self = .gemma4(filter)
                return output

            case .longCat(var filter):
                let output = filter.feed(text)
                self = .longCat(filter)
                return output

            case .miniMaxM3(var filter):
                let output = filter.feed(text)
                self = .miniMaxM3(filter)
                return output
            }
        }

        mutating func finish() -> String {
            switch self {
            case .cohere(var filter):
                let output = filter.finish()
                self = .cohere(filter)
                return output

            case .harmony(var filter):
                let output = filter.finish()
                self = .harmony(filter)
                return output

            case .gemma4(var filter):
                let output = filter.finish()
                self = .gemma4(filter)
                return output

            case .longCat(var filter):
                let output = filter.finish()
                self = .longCat(filter)
                return output

            case .miniMaxM3(var filter):
                let output = filter.finish()
                self = .miniMaxM3(filter)
                return output
            }
        }
    }

    private var stages: [Stage]

    init(promptStyle: MLXPromptStyle? = nil) {
        stages = Self.stages(for: promptStyle)
    }

    mutating func feed(_ text: String) -> String {
        stages.indices.reduce(text) { output, index in
            stages[index].feed(output)
        }
    }

    mutating func finish() -> String {
        guard !stages.isEmpty else {
            return ""
        }

        var output = stages[stages.startIndex].finish()
        for index in stages.indices.dropFirst() {
            output = stages[index].feed(output) + stages[index].finish()
        }
        return output
    }

    private static func stages(for promptStyle: MLXPromptStyle?) -> [Stage] {
        switch promptStyle {
        case .some(.cohereAction):
            return [.cohere(MLXCohereCommandStreamFilter())]

        case .some(.harmony):
            return [.harmony(MLXHarmonyStreamFilter())]

        case .some(.gemma):
            return [.gemma4(MLXGemma4StreamFilter())]

        case .some(.longCat):
            return [.longCat(MLXLongCatStreamFilter())]

        case .some(.minimaxM3):
            return [.miniMaxM3(MLXMiniMaxM3StreamFilter())]

        case nil:
            return [
                .cohere(MLXCohereCommandStreamFilter()),
                .harmony(MLXHarmonyStreamFilter()),
                .gemma4(MLXGemma4StreamFilter()),
                .longCat(MLXLongCatStreamFilter()),
                .miniMaxM3(MLXMiniMaxM3StreamFilter())
            ]

        default:
            return []
        }
    }
}
