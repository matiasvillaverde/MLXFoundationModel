import Foundation

/// Errors raised by ``MLXModelPool`` registration and residency management.
public enum MLXModelPoolError: Error, Equatable, LocalizedError, Sendable {
    case aliasAlreadyRegistered(alias: String, existingModelID: String)
    case capacityExhausted(maxResidentModels: Int)
    case duplicateModel(String)
    case duplicateProfile(String)
    case invalidProfileName(modelID: String)
    case residentMemoryCapacityExhausted(
        requestedBytes: Int,
        maxResidentMemoryBytes: Int,
        residentBytes: Int
    )
    case unknownModel(String)

    /// Human-readable error description.
    public var errorDescription: String? {
        switch self {
        case let .aliasAlreadyRegistered(alias, existingModelID):
            "Alias '\(alias)' already points to '\(existingModelID)'."

        case let .capacityExhausted(maxResidentModels):
            "No evictable MLX model is available within the resident model limit \(maxResidentModels)."

        case let .duplicateModel(id):
            "An MLX model is already registered for '\(id)'."

        case let .duplicateProfile(id):
            "An MLX model serving profile is already registered for '\(id)'."

        case let .invalidProfileName(modelID):
            "The serving profile for '\(modelID)' must have a non-empty name."

        case let .residentMemoryCapacityExhausted(requestedBytes, maxBytes, residentBytes):
            "No evictable MLX model can free enough memory for \(Self.format(requestedBytes)); " +
                "resident models use \(Self.format(residentBytes)) of the \(Self.format(maxBytes)) budget."

        case let .unknownModel(id):
            "No MLX model is registered for '\(id)'."
        }
    }

    private static func format(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}
