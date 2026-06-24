import Foundation

/// Synchronous, thread-safe lookup table for known model configurations.
internal class AbstractModelRegistry: @unchecked Sendable {
    internal init() {
        self.configurationsByName = [:]
    }

    internal init(modelConfigurations: [ModelConfiguration]) {
        self.configurationsByName = Self.dictionary(from: modelConfigurations)
    }

    private let lock = NSLock()
    private var configurationsByName: [String: ModelConfiguration]

    internal func register(configurations: [ModelConfiguration]) {
        lock.withLock {
            for configuration in configurations {
                configurationsByName[configuration.name] = configuration
            }
        }
    }

    internal func configuration(id: String) -> ModelConfiguration {
        lock.withLock {
            configurationsByName[id] ?? ModelConfiguration(id: id)
        }
    }

    internal func contains(id: String) -> Bool {
        lock.withLock {
            configurationsByName[id] != nil
        }
    }

    internal var models: [ModelConfiguration] {
        lock.withLock {
            Array(configurationsByName.values)
        }
    }

    private static func dictionary(
        from configurations: [ModelConfiguration]
    ) -> [String: ModelConfiguration] {
        Dictionary(uniqueKeysWithValues: configurations.map { ($0.name, $0) })
    }
}
