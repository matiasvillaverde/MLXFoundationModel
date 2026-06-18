import Foundation

internal struct GenerationConfigFile: Decodable, Sendable {
    internal let eosTokenIds: IntOrIntArray?

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
    }
}
