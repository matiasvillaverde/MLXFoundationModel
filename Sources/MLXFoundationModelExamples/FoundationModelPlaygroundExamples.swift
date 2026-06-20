import Foundation
import MLXFoundationModel
import MLXLocalModels

/// Reusable prompt shapes mirrored by the runnable playground and real-model tests.
public enum FoundationModelPlaygroundExamples {
    public static let fruitChoices = ["apple", "pear", "banana"]

    public static var all: [FoundationModelPlaygroundExample] {
        [
            streamingChat,
            tripPlannerGuidedGeneration,
            pointsOfInterestToolCalling,
            finiteChoiceGuidedGeneration,
            contentTagging
        ]
    }

    public static let streamingChat = FoundationModelPlaygroundExample(
        id: "streaming-chat",
        title: "Streaming chat",
        request: MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nWrite one short sentence about local MLX inference."
                )
            ],
            instructions: "You are a concise assistant. Do not think aloud."
        ),
        sampling: .deterministic,
        limits: ResourceLimits(maxTokens: 32, maxTime: .seconds(120), reusePromptCache: false)
    )

    public static let tripPlannerGuidedGeneration = FoundationModelPlaygroundExample(
        id: "apple-trip-planner-guided-generation",
        title: "Apple Trip Planner guided generation",
        request: MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: """
                    Generate a one-day itinerary to Yosemite. Give it a fun title, \
                    choose the Yosemite destination, and include one activity.
                    """
                )
            ],
            instructions: """
            Your job is to create an itinerary for the person.
            Each day needs an activity, hotel, or restaurant.
            Here is an example, but do not copy it: title Onsen Trip to Japan, \
            destination Mt. Fuji, activity sightseeing.
            """,
            responseConstraint: MLXBridgeResponseConstraint(
                jsonSchema: tripPlannerSchema,
                instructions: "Return only a structured itinerary matching this schema."
            )
        ),
        sampling: SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .jsonSchema(tripPlannerSchema))
        ),
        limits: ResourceLimits(maxTokens: 128, maxTime: .seconds(120), reusePromptCache: false)
    )

    public static let pointsOfInterestToolCalling = FoundationModelPlaygroundExample(
        id: "apple-points-of-interest-tool-calling",
        title: "Apple points-of-interest tool calling",
        request: MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "Call the findPointsOfInterest tool for a hotel near Yosemite."
                )
            ],
            instructions: """
            You are a tool router. Return exactly one tool-call JSON object and no prose.
            """,
            tools: [
                MLXBridgeToolDefinition(
                    name: "findPointsOfInterest",
                    description: "Finds points of interest for a landmark.",
                    parametersJSONSchema: pointsOfInterestArgumentsSchema
                )
            ]
        ),
        sampling: SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .jsonSchema(pointsOfInterestToolCallSchema))
        ),
        limits: ResourceLimits(maxTokens: 80, maxTime: .seconds(120), reusePromptCache: false)
    )

    public static let finiteChoiceGuidedGeneration = FoundationModelPlaygroundExample(
        id: "apple-finite-choice-guided-generation",
        title: "Apple finite-choice guided generation",
        request: MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "Choose one fruit. Do not write anything except the fruit."
                )
            ],
            instructions: "Choose exactly one allowed fruit."
        ),
        sampling: SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .choices(fruitChoices))
        ),
        limits: ResourceLimits(maxTokens: 8, maxTime: .seconds(120), reusePromptCache: false)
    )

    public static let contentTagging = FoundationModelPlaygroundExample(
        id: "apple-content-tagging",
        title: "Apple content-tagging guided generation",
        request: MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: """
                    Yosemite has granite cliffs, waterfalls, hiking trails, \
                    forests, and scenic viewpoints.
                    """
                )
            ],
            instructions: "Tag the three most important topics in the input text.",
            responseConstraint: MLXBridgeResponseConstraint(
                jsonSchema: contentTaggingSchema,
                instructions: "Return only topic tags matching this schema."
            )
        ),
        sampling: SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(grammar: .jsonSchema(contentTaggingSchema))
        ),
        limits: ResourceLimits(maxTokens: 80, maxTime: .seconds(120), reusePromptCache: false)
    )

    public static let tripPlannerSchema = """
    {"type":"object","properties":{"title":{"type":"string"},"destinationName":{"enum":["Yosemite"]},\
    "day":{"type":"object","properties":{"title":{"type":"string"},"activityKind":\
    {"enum":["sightseeing","foodAndDining","hotelAndLodging"]},"activity":{"type":"string"}},\
    "required":["title","activityKind","activity"],"additionalProperties":false}},\
    "required":["title","destinationName","day"],"additionalProperties":false}
    """

    public static let pointsOfInterestArgumentsSchema = """
    {"type":"object","properties":{"pointOfInterest":{"enum":["hotel"]},\
    "naturalLanguageQuery":{"enum":["hotel near Yosemite"]}},\
    "required":["pointOfInterest","naturalLanguageQuery"],"additionalProperties":false}
    """

    public static let pointsOfInterestToolCallSchema = """
    {"type":"object","properties":{"tool_name":{"enum":["findPointsOfInterest"]},\
    "arguments":\(pointsOfInterestArgumentsSchema)},\
    "required":["tool_name","arguments"],"additionalProperties":false}
    """

    public static let contentTaggingSchema = """
    {"type":"object","properties":{"tags":{"type":"array","minItems":3,"maxItems":3,\
    "items":{"type":"string"}}},"required":["tags"],"additionalProperties":false}
    """
}

public struct FoundationModelPlaygroundExample: Hashable, Sendable {
    public let id: String
    public let title: String
    public let request: MLXBridgeRequest
    public let style: MLXPromptStyle
    public let sampling: SamplingParameters
    public let limits: ResourceLimits

    public init(
        id: String,
        title: String,
        request: MLXBridgeRequest,
        style: MLXPromptStyle = .chatML,
        sampling: SamplingParameters,
        limits: ResourceLimits
    ) {
        self.id = id
        self.title = title
        self.request = request
        self.style = style
        self.sampling = sampling
        self.limits = limits
    }

    public func resolvedStyle(modelDefault: MLXPromptStyle) -> MLXPromptStyle {
        guard style == .chatML else {
            return style
        }
        return modelDefault == .plain ? .chatML : modelDefault
    }
}
