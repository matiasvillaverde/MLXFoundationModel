import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument conditional schemas")
struct MLXToolArgumentConditionalSchemaTests {
    @Test("normalizes arguments with matching then branch schemas")
    func normalizesArgumentsWithMatchingThenBranchSchemas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=route>\
            <parameter=mode>search</parameter>\
            <parameter=query>123</parameter>\
            <parameter=limit>"4"</parameter>\
            <parameter=path>drop me</parameter>\
            </function></tool_call>
            """,
            tools: [Self.routeTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["mode"] as? String == "search")
        #expect(arguments["query"] as? String == "123")
        #expect(arguments["limit"] as? Int == 4)
        #expect(arguments["path"] == nil)
    }

    @Test("normalizes arguments with matching else branch schemas")
    func normalizesArgumentsWithMatchingElseBranchSchemas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=route>\
            <parameter=mode>open</parameter>\
            <parameter=path>123</parameter>\
            <parameter=limit>"4"</parameter>\
            </function></tool_call>
            """,
            tools: [Self.routeTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["mode"] as? String == "open")
        #expect(arguments["path"] as? String == "123")
        #expect(arguments["limit"] == nil)
    }

    @Test("normalizes ref backed dependentSchemas when trigger property is present")
    func normalizesReferenceBackedDependentSchemas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=configure>\
            <parameter=provider>mlx</parameter>\
            <parameter=token>12345</parameter>\
            <parameter=timeout>"30"</parameter>\
            <parameter=extra>drop me</parameter>\
            </function></tool_call>
            """,
            tools: [Self.dependentSchemaTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["provider"] as? String == "mlx")
        #expect(arguments["token"] as? String == "12345")
        #expect(arguments["timeout"] as? Int == 30)
        #expect(arguments["extra"] == nil)
    }

    @Test("does not apply dependentSchemas when trigger property is absent")
    func doesNotApplyDependentSchemaWithoutTriggerProperty() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=configure>\
            <parameter=token>12345</parameter>\
            <parameter=timeout>"30"</parameter>\
            </function></tool_call>
            """,
            tools: [Self.dependentSchemaTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["token"] == nil)
        #expect(arguments["timeout"] == nil)
    }

    @Test("selects then branch only when dependentRequired matches")
    func selectsThenBranchOnlyWhenDependentRequiredMatches() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=connect>\
            <parameter=provider>cloud</parameter>\
            <parameter=region>eu</parameter>\
            <parameter=timeout>"30"</parameter>\
            <parameter=offline>yes</parameter>\
            </function></tool_call>
            """,
            tools: [Self.dependentRequiredTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["provider"] as? String == "cloud")
        #expect(arguments["region"] as? String == "eu")
        #expect(arguments["timeout"] as? Int == 30)
        #expect(arguments["offline"] == nil)
    }

    @Test("selects else branch when dependentRequired is missing")
    func selectsElseBranchWhenDependentRequiredIsMissing() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=connect>\
            <parameter=provider>cloud</parameter>\
            <parameter=timeout>"30"</parameter>\
            <parameter=offline>yes</parameter>\
            </function></tool_call>
            """,
            tools: [Self.dependentRequiredTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["provider"] as? String == "cloud")
        #expect(arguments["region"] == nil)
        #expect(arguments["timeout"] == nil)
        #expect(arguments["offline"] as? Bool == true)
    }

    @Test("selects then branch when array contains schema matches")
    func selectsThenBranchWhenArrayContainsSchemaMatches() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=plan>\
            <parameter=modes>["fast","advanced"]</parameter>\
            <parameter=depth>"3"</parameter>\
            <parameter=simple>yes</parameter>\
            </function></tool_call>
            """,
            tools: [Self.arrayContainsTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let modes = try #require(arguments["modes"] as? [String])

        #expect(modes == ["fast", "advanced"])
        #expect(arguments["depth"] as? Int == 3)
        #expect(arguments["simple"] == nil)
    }

    @Test("selects else branch when array contains schema does not match")
    func selectsElseBranchWhenArrayContainsSchemaDoesNotMatch() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=plan>\
            <parameter=modes>["fast","safe"]</parameter>\
            <parameter=depth>"3"</parameter>\
            <parameter=simple>yes</parameter>\
            </function></tool_call>
            """,
            tools: [Self.arrayContainsTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let modes = try #require(arguments["modes"] as? [String])

        #expect(modes == ["fast", "safe"])
        #expect(arguments["depth"] == nil)
        #expect(arguments["simple"] as? Bool == true)
    }

    private static var routeTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "route",
            description: "Route a request",
            parametersJSONSchema: Self.routeSchema
        )
    }

    private static var dependentSchemaTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "configure",
            description: "Configure a provider",
            parametersJSONSchema: Self.dependentSchema
        )
    }

    private static var dependentRequiredTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "connect",
            description: "Connect to a provider",
            parametersJSONSchema: Self.dependentRequiredSchema
        )
    }

    private static var arrayContainsTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "plan",
            description: "Plan a request",
            parametersJSONSchema: Self.arrayContainsSchema
        )
    }

    private static var routeSchema: String {
        [
            #"{"type":"object","additionalProperties":false,"#,
            #""properties":{"mode":{"enum":["search","open"]}},"#,
            #""if":{"required":["mode"],"properties":{"mode":{"const":"search"}}},"#,
            #""then":{"properties":{"query":{"type":"string"},"limit":{"type":"integer"}}},"#,
            #""else":{"properties":{"path":{"type":"string"}}}}"#
        ].joined()
    }

    private static var dependentSchema: String {
        [
            #"{"type":"object","additionalProperties":false,"$defs":{"#,
            #""ProviderSettings":{"properties":{"token":{"type":"string"},"#,
            #""timeout":{"type":"integer"}}}},"#,
            #""properties":{"provider":{"type":"string"}},"#,
            ##""dependentSchemas":{"provider":{"$ref":"#/$defs/ProviderSettings"}}}"##
        ].joined()
    }

    private static var dependentRequiredSchema: String {
        [
            #"{"type":"object","additionalProperties":false,"#,
            #""properties":{"provider":{"const":"cloud"},"region":{"type":"string"}},"#,
            #""if":{"required":["provider"],"dependentRequired":{"provider":["region"]},"#,
            #""properties":{"provider":{"const":"cloud"}}},"#,
            #""then":{"properties":{"timeout":{"type":"integer"}}},"#,
            #""else":{"properties":{"offline":{"type":"boolean"}}}}"#
        ].joined()
    }

    private static var arrayContainsSchema: String {
        [
            #"{"type":"object","additionalProperties":false,"#,
            #""properties":{"modes":{"type":"array","items":{"type":"string"}}},"#,
            #""if":{"required":["modes"],"properties":{"modes":{"type":"array","#,
            #""contains":{"const":"advanced"},"minContains":1,"maxContains":1}}},"#,
            #""then":{"properties":{"depth":{"type":"integer"}}},"#,
            #""else":{"properties":{"simple":{"type":"boolean"}}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
