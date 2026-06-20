import Foundation

enum MLXSafetensorsHeaderScanner {
    private static let maxHeaderBytes = 64 * 1_024 * 1_024

    static func tensors(in modelDirectory: URL) throws -> [MLXSafetensorsHeaderTensor] {
        let urls = try safetensorsURLs(in: modelDirectory)
        return try urls.flatMap { url in
            try tensors(at: url)
        }
    }

    private static func safetensorsURLs(in modelDirectory: URL) throws -> [URL] {
        let urls = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        return urls
            .filter { $0.pathExtension == "safetensors" }
            .sorted { left, right in left.lastPathComponent < right.lastPathComponent }
    }

    private static func tensors(at url: URL) throws -> [MLXSafetensorsHeaderTensor] {
        let headerData = try headerData(at: url)
        guard let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw MLXOQModelArtifactPlannerError.invalidSafetensorsHeader(url)
        }
        return object.compactMap { key, value in
            tensor(name: key, sourceFilename: url.lastPathComponent, value: value)
        }
    }

    private static func headerData(at url: URL) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8),
            lengthData.count == 8 else {
            throw MLXOQModelArtifactPlannerError.emptySafetensorsHeader(url)
        }
        let headerLength = littleEndianUInt64(lengthData)
        guard headerLength <= UInt64(maxHeaderBytes),
            let headerData = try handle.read(upToCount: Int(headerLength)),
            headerData.count == Int(headerLength) else {
            throw MLXOQModelArtifactPlannerError.invalidSafetensorsHeader(url)
        }
        return headerData
    }

    private static func tensor(
        name: String,
        sourceFilename: String,
        value: Any
    ) -> MLXSafetensorsHeaderTensor? {
        guard name != "__metadata__",
            let metadata = value as? [String: Any] else {
            return nil
        }
        return MLXSafetensorsHeaderTensor(
            dtype: metadata["dtype"] as? String,
            name: name,
            shape: intArray(metadata["shape"]),
            sourceFilename: sourceFilename
        )
    }

    private static func intArray(_ value: Any?) -> [Int] {
        guard let values = value as? [Any] else {
            return []
        }
        return values.compactMap { value in
            if let int = value as? Int {
                return int
            }
            if let double = value as? Double {
                return Int(double)
            }
            return nil
        }
    }

    private static func littleEndianUInt64(_ data: Data) -> UInt64 {
        data.enumerated().reduce(UInt64(0)) { value, element in
            let (offset, byte) = element
            return value | (UInt64(byte) << UInt64(offset * 8))
        }
    }
}
