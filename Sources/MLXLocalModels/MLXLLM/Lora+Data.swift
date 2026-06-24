import Foundation

internal enum LoRADataError: LocalizedError, Equatable {
    case fileNotFound(directory: URL, name: String)
    case unsupportedFileExtension(URL)

    internal var errorDescription: String? {
        switch self {
        case .fileNotFound(let directory, let name):
            "Could not find LoRA data file '\(name)' in '\(directory.path())'."
        case .unsupportedFileExtension(let url):
            "Unsupported LoRA data file extension: '\(url.pathExtension)'."
        }
    }
}

internal enum LoRADataFormat: String, CaseIterable, Sendable {
    case jsonLines = "jsonl"
    case text = "txt"

    internal init?(url: URL) {
        self.init(rawValue: url.pathExtension.lowercased())
    }
}

internal func loadLoRAData(directory: URL, name: String) throws -> [String] {
    for format in LoRADataFormat.allCases {
        let candidateURL = directory.appending(component: "\(name).\(format.rawValue)")
        guard FileManager.default.fileExists(atPath: candidateURL.path()) else {
            continue
        }
        return try loadLoRAData(url: candidateURL)
    }

    throw LoRADataError.fileNotFound(directory: directory, name: name)
}

internal func loadLoRAData(url: URL) throws -> [String] {
    guard let format = LoRADataFormat(url: url) else {
        throw LoRADataError.unsupportedFileExtension(url)
    }

    switch format {
    case .jsonLines:
        return try loadJSONL(url: url)
    case .text:
        return try loadLines(url: url)
    }
}

internal func loadJSONL(url: URL) throws -> [String] {
    struct Record: Decodable {
        let text: String?
    }

    let decoder = JSONDecoder()
    let lines = try loadRawLines(url: url)

    return try lines.compactMap { line in
        let trimmedLine = line.trimmingCharacters(in: .whitespaces)
        guard trimmedLine.first == "{" else {
            return nil
        }

        return try decoder.decode(Record.self, from: Data(trimmedLine.utf8)).text
    }
}

internal func loadLines(url: URL) throws -> [String] {
    try loadRawLines(url: url).filter { !$0.isEmpty }
}

private func loadRawLines(url: URL) throws -> [String] {
    try String(contentsOf: url, encoding: .utf8)
        .components(separatedBy: .newlines)
}
