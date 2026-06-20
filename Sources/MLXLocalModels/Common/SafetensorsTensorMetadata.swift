import Foundation

internal enum MLXSafetensorsTensorMetadata {
    private static let maxHeaderBytes = 64 * 1_024 * 1_024
    private static let dtypeMetadataPrefix = "__mlx_tensor_dtype."

    internal static func dtypeMetadataKey(_ tensorName: String) -> String {
        dtypeMetadataPrefix + tensorName
    }

    internal static func dtype(
        for tensorName: String,
        in metadata: [String: String]
    ) -> String? {
        metadata[dtypeMetadataKey(tensorName)]
    }

    internal static func encodedDTypeMetadata(at url: URL) -> [String: String] {
        tensorDTypes(at: url).reduce(into: [String: String]()) { result, entry in
            result[dtypeMetadataKey(entry.key)] = entry.value
        }
    }

    private static func tensorDTypes(at url: URL) -> [String: String] {
        guard let headerData = safetensorsHeaderData(at: url) else {
            return [:]
        }
        guard let object = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return [:]
        }
        return object.reduce(into: [String: String]()) { result, entry in
            guard entry.key != "__metadata__",
                let metadata = entry.value as? [String: Any],
                let dtype = metadata["dtype"] as? String
            else {
                return
            }
            result[entry.key] = dtype
        }
    }

    private static func safetensorsHeaderData(at url: URL) -> Data? {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return nil
        }
        defer { try? handle.close() }

        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else {
            return nil
        }
        let headerLength = littleEndianUInt64(lengthData)
        guard headerLength <= UInt64(maxHeaderBytes) else {
            return nil
        }
        guard let headerData = try? handle.read(upToCount: Int(headerLength)),
            headerData.count == Int(headerLength)
        else {
            return nil
        }
        return headerData
    }

    private static func littleEndianUInt64(_ data: Data) -> UInt64 {
        data.enumerated().reduce(UInt64(0)) { value, element in
            let (offset, byte) = element
            return value | (UInt64(byte) << UInt64(offset * 8))
        }
    }
}
