import Foundation

enum MLXModelWeightArtifactScanner {
    private static let maxSafetensorsHeaderBytes = 64 * 1_024 * 1_024
    private static let fp8ScaleSuffix = ".scale"
    private static let fp8ScaleInvSuffix = "_scale_inv"
    private static let weightSuffix = ".weight"
    private static let residentWeightExtensions: Set<String> = [
        "bin",
        "gguf",
        "npz",
        "safetensors"
    ]
    private static let mtpWeightPrefixes = [
        "mtp.",
        "language_model.mtp.",
        "model.mtp.",
        "model.language_model.mtp."
    ]
    private static let fp8WeightDTypes: Set<String> = ["F8_E4M3", "F8_E5M2", "I8"]

    private struct TensorMetadata: Equatable {
        let name: String
        let dtype: String?
        let shape: [Int]
    }

    struct Evidence: Equatable, Sendable {
        let hasMTPWeightTensors: Bool
        let hasFP8ScaleSidecars: Bool
    }

    static func evidence(in modelDirectory: URL) -> Evidence {
        let indexKeys = safetensorsIndexKeys(
            at: modelDirectory.appendingPathComponent("model.safetensors.index.json")
        )
        let headerMetadata = safetensorsHeaderMetadata(in: modelDirectory)
        let headerKeys = headerMetadata.map(\.name)
        let tensorKeys = indexKeys + headerKeys
        return Evidence(
            hasMTPWeightTensors: tensorKeys.contains(where: isMTPWeightName),
            hasFP8ScaleSidecars: hasFP8ScaleSidecars(
                indexKeys: indexKeys,
                headerMetadata: headerMetadata
            )
        )
    }

    static func hasMTPWeightTensors(in modelDirectory: URL) -> Bool {
        evidence(in: modelDirectory).hasMTPWeightTensors
    }

    static func hasFP8ScaleSidecars(in modelDirectory: URL) -> Bool {
        evidence(in: modelDirectory).hasFP8ScaleSidecars
    }

    static func residentWeightBytes(in modelDirectory: URL) -> Int {
        weightArtifactURLs(in: modelDirectory).reduce(0) { total, url in
            total + fileSize(at: url)
        }
    }

    private static func weightArtifactURLs(in modelDirectory: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }
        return enumerator.compactMap { item in
            guard let url = item as? URL, isResidentWeightArtifact(url) else {
                return nil
            }
            return url
        }
    }

    private static func isResidentWeightArtifact(_ url: URL) -> Bool {
        residentWeightExtensions.contains(url.pathExtension.lowercased())
    }

    private static func fileSize(at url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    }

    private static func safetensorsIndexKeys(at url: URL) -> [String] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        guard let data = try? Data(contentsOf: url) else {
            return []
        }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }
        guard let weightMap = object["weight_map"] as? [String: Any] else {
            return []
        }
        return Array(weightMap.keys)
    }

    private static func safetensorsHeaderMetadata(in modelDirectory: URL) -> [TensorMetadata] {
        guard let filenames = try? FileManager.default.contentsOfDirectory(atPath: modelDirectory.path) else {
            return []
        }

        return filenames
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
            .flatMap { filename in
                safetensorsHeaderMetadata(at: modelDirectory.appendingPathComponent(filename))
            }
    }

    private static func safetensorsHeaderMetadata(at url: URL) -> [TensorMetadata] {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return []
        }
        defer { try? handle.close() }

        guard let lengthData = try? handle.read(upToCount: 8) else {
            return []
        }
        guard lengthData.count == 8 else {
            return []
        }

        let headerLength = littleEndianUInt64(lengthData)
        guard headerLength <= UInt64(maxSafetensorsHeaderBytes) else {
            return []
        }
        guard let headerData = try? handle.read(upToCount: Int(headerLength)) else {
            return []
        }
        guard headerData.count == Int(headerLength) else {
            return []
        }
        return parsedTensorMetadata(from: headerData)
    }

    private static func littleEndianUInt64(_ data: Data) -> UInt64 {
        data.enumerated().reduce(UInt64(0)) { value, element in
            let (offset, byte) = element
            return value | (UInt64(byte) << UInt64(offset * 8))
        }
    }

    private static func isMTPWeightName(_ name: String) -> Bool {
        mtpWeightPrefixes.contains { name.hasPrefix($0) }
    }

    private static func parsedTensorMetadata(from headerData: Data) -> [TensorMetadata] {
        guard let object = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return []
        }
        return object.compactMap { key, value in
            tensorMetadata(key: key, value: value)
        }
    }

    private static func tensorMetadata(
        key: String,
        value: Any
    ) -> TensorMetadata? {
        guard key != "__metadata__" else {
            return nil
        }
        let metadata = value as? [String: Any]
        return TensorMetadata(
            name: key,
            dtype: metadata?["dtype"] as? String,
            shape: intArray(metadata?["shape"])
        )
    }

    private static func hasFP8ScaleSidecars(
        indexKeys: [String],
        headerMetadata: [TensorMetadata]
    ) -> Bool {
        if indexKeys.contains(where: isFP8ScaleInvSidecarName) {
            return true
        }
        if headerMetadata.contains(where: { isFP8ScaleInvSidecarName($0.name) }) {
            return true
        }
        return hasHeaderFP8ScalePair(headerMetadata)
    }

    private static func hasHeaderFP8ScalePair(_ metadata: [TensorMetadata]) -> Bool {
        var metadataByName = [String: TensorMetadata]()
        for item in metadata {
            metadataByName[item.name] = item
        }
        return metadata.contains { item in
            guard let weightName = fp8WeightName(forScaleName: item.name),
                let weight = metadataByName[weightName]
            else {
                return false
            }
            return isFP8WeightDType(weight.dtype) && isMatrixPair(weight: weight, scale: item)
        }
    }

    private static func fp8WeightName(forScaleName name: String) -> String? {
        guard name.hasSuffix(fp8ScaleSuffix) else {
            return nil
        }
        return String(name.dropLast(fp8ScaleSuffix.count)) + weightSuffix
    }

    private static func isFP8ScaleInvSidecarName(_ name: String) -> Bool {
        name.hasSuffix(fp8ScaleInvSuffix)
    }

    private static func isFP8WeightDType(_ dtype: String?) -> Bool {
        guard let dtype else {
            return false
        }
        return fp8WeightDTypes.contains(dtype.uppercased())
    }

    private static func isMatrixPair(weight: TensorMetadata, scale: TensorMetadata) -> Bool {
        weight.shape.count == 2 && scale.shape.count == 2
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
}
