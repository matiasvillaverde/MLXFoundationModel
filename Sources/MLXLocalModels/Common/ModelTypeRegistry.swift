import Foundation

/// Maps a `config.json` model type to a model constructor.
internal class ModelTypeRegistry: @unchecked Sendable {
    internal typealias Creator = @Sendable (URL) throws -> any LanguageModel

    internal init() {
        self.creatorsByType = [:]
    }

    internal init(creators: [String: Creator]) {
        self.creatorsByType = creators
    }

    private let lock = NSLock()
    private var creatorsByType: [String: Creator]

    internal func registerModelType(
        _ type: String,
        creator: @escaping Creator
    ) {
        lock.withLock {
            creatorsByType[type] = creator
        }
    }

    internal func createModel(
        configuration: URL,
        modelType: String
    ) throws -> any LanguageModel {
        let creator = lock.withLock {
            creatorsByType[modelType]
        }
        guard let creator else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

    internal func registeredModelTypes() -> [String] {
        lock.withLock {
            creatorsByType.keys.sorted()
        }
    }
}
