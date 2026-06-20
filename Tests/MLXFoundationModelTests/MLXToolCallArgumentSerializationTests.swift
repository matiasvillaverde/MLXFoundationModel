import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool call argument serialization")
struct MLXToolCallArgumentSerializationTests {
    @Test("decodes JSON object strings emitted as tool arguments")
    func decodesJSONObjectStringsEmittedAsToolArguments() throws {
        let calls = MLXToolCallExtractor.extractAll(
            from: #"""
            {
                "tool_calls": [{
                    "function": {
                        "name": "weather",
                        "arguments": "{\"city\":\"Berlin\",\"count\":2}"
                    }
                }]
            }
            """#
        )
        let call = try #require(calls.first)

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin","count":2}"#)
    }

    @Test("coerces non-object tool arguments to an empty object")
    func coercesNonObjectToolArgumentsToAnEmptyObject() throws {
        let flat = try #require(MLXToolCallExtractor.extract(
            from: #"{"name":"weather","arguments":["Berlin"]}"#
        ))
        let string = try #require(MLXToolCallExtractor.extract(
            from: #"{"name":"weather","arguments":"not json"}"#
        ))

        #expect(flat.argumentsJSON == "{}")
        #expect(string.argumentsJSON == "{}")
    }

    @Test("coerces MiniMax M3 non-object tool arguments to an empty object")
    func coercesMiniMaxM3NonObjectToolArgumentsToAnEmptyObject() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            ]<]minimax[>[<tool_call>
            ]<]minimax[>[<invoke name="weather">\
            ]<]minimax[>[<item>Berlin]<]minimax[>[</item>\
            ]<]minimax[>[</invoke>
            ]<]minimax[>[</tool_call>
            """
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == "{}")
    }
}
