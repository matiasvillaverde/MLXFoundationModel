#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
extension FMRequiredToolGrammarBuilder {
    static func escapedEBNFLiteral(_ value: String) -> String {
        value.reduce(into: "") { result, character in
            switch character {
            case "\"":
                result += #"\""#

            case "\\":
                result += #"\\"#

            case "\n":
                result += #"\n"#

            case "\r":
                result += #"\r"#

            case "\t":
                result += #"\t"#

            default:
                result.append(character)
            }
        }
    }
}
#endif
