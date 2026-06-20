#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
extension FMRequiredToolGrammarBuilder {
    static func stringEncodedParameterValue(
        _ schema: [String: Any],
        format: FMNativeToolGrammarFormat
    ) -> Bool {
        guard format == .deepSeekDSML else {
            return true
        }
        if let constant = schema["const"] {
            return constant is String
        }
        if let values = schema["enum"] as? [Any], !values.isEmpty {
            return values.allSatisfy { $0 is String }
        }
        let types = schemaTypes(schema)
        guard !types.isEmpty else {
            return true
        }
        return types.allSatisfy { $0 == "string" || $0 == "null" }
    }

    private static func schemaTypes(_ schema: [String: Any]) -> [String] {
        if let type = schema["type"] as? String {
            return [type]
        }
        if let types = schema["type"] as? [String] {
            return types
        }
        return []
    }
}
#endif
