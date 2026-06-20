import Foundation

internal struct GenerationConfigFile: Decodable, Sendable {
    internal let eosTokenIds: IntOrIntArray?
    internal let suppressTokenIds: IntOrIntArray?

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case suppressTokenIds = "suppress_tokens"
    }
}
