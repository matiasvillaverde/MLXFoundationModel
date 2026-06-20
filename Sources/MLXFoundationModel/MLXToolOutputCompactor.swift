import Foundation

enum MLXToolOutputCompactor {
    static let defaultCharacterLimit = 16_384

    static func compact(
        _ text: String,
        limit: Int = defaultCharacterLimit
    ) -> String {
        guard limit > 0 else {
            return ""
        }

        let count = text.count
        guard count > limit else {
            return text
        }

        var markerText = marker(omittedCharacters: count)
        var previousMarkerLength: Int?
        while markerText.count != previousMarkerLength {
            previousMarkerLength = markerText.count
            let available = max(0, limit - markerText.count)
            let prefixCount = available / 2
            let suffixCount = available - prefixCount
            markerText = marker(omittedCharacters: count - prefixCount - suffixCount)
        }

        let available = max(0, limit - markerText.count)
        guard available > 0 else {
            return String(markerText.prefix(limit))
        }

        let prefixCount = available / 2
        let suffixCount = available - prefixCount
        return String(text.prefix(prefixCount))
            + markerText
            + String(text.suffix(suffixCount))
    }

    private static func marker(omittedCharacters: Int) -> String {
        "\n\n[MLXFoundationModel truncated \(omittedCharacters) characters from this tool output]\n\n"
    }
}
