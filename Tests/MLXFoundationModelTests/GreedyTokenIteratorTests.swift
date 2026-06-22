import MLX
@testable import MLXLocalModels
import Testing

@Suite("Greedy token iterator fast path")
struct GreedyTokenIteratorTests {
    @Test("uses greedy token model when argmax has no processor")
    func usesGreedyTokenModelForPlainArgmax() throws {
        let model = ScriptedGreedyTokenModel()
        let input = LMInput(text: .init(tokens: MLXArray([1])))
        var iterator = try TokenIterator(
            input: input,
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )

        #expect(iterator.next() == 7)
        #expect(model.greedyStepCount == 2)
        #expect(model.fullLogitsStepCount == 0)
    }
}
