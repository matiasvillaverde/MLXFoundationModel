import Foundation

enum MLXRealModelCatalog {
    struct Model: Codable, Hashable, Sendable {
        let id: String
        let displayName: String
        let architecture: String
        let repository: String?
        let relativePath: String
        let prompt: String
        let expectedTokens: [String]
        let maxTokens: Int
        let tags: [String]

        var isDownloadable: Bool {
            repository != nil
        }
    }

    static func load() throws -> [Model] {
        let url = try catalogURL()
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([Model].self, from: data)
    }

    private static func catalogURL() throws -> URL {
        if let url = Bundle.module.url(
            forResource: "model-catalog",
            withExtension: "json",
            subdirectory: "Resources"
        ) {
            return url
        }
        throw CocoaError(.fileNoSuchFile)
    }
}
