import Foundation

extension MLXToolCallExtractor {
    static func extractHarmonyToolCalls(from text: String) -> [MLXExtractedToolCall] {
        guard text.contains("<|channel|>"), text.contains("<|message|>") else {
            return []
        }

        var calls: [MLXExtractedToolCall] = []
        var searchStart = text.startIndex
        while let messageRange = text.range(of: "<|message|>", range: searchStart..<text.endIndex) {
            guard let call = harmonyCall(
                in: text,
                searchStart: searchStart,
                messageRange: messageRange
            ) else {
                searchStart = messageRange.upperBound
                continue
            }
            guard let contentEnd = firstHarmonyTerminator(in: text, after: messageRange.upperBound) else {
                break
            }
            calls.append(MLXExtractedToolCall(
                name: call.name,
                argumentsJSON: Parser.canonicalArgumentsJSONString(Parser.decodedArgumentValue(
                    String(text[messageRange.upperBound..<contentEnd.lowerBound])
                ))
            ))
            searchStart = contentEnd.upperBound
        }
        return calls
    }

    private static func harmonyCall(
        in text: String,
        searchStart: String.Index,
        messageRange: Range<String.Index>
    ) -> (name: String, channel: String)? {
        let header = String(text[searchStart..<messageRange.lowerBound])
        guard let channelRange = header.range(of: "<|channel|>", options: .backwards) else {
            return nil
        }
        let leadingHeader = String(header[..<channelRange.lowerBound])
        let channelDescriptor = String(header[channelRange.upperBound...])
        guard MLXHarmonyStreamFilter.channelName(in: channelDescriptor) == "commentary" else {
            return nil
        }
        let recipient = MLXHarmonyStreamFilter.recipient(in: leadingHeader)
            ?? MLXHarmonyStreamFilter.recipient(in: channelDescriptor)
        guard let recipient else {
            return nil
        }
        let name = recipient.hasPrefix("functions.")
            ? String(recipient.dropFirst("functions.".count))
            : recipient
        return (name, channelDescriptor)
    }

    private static func firstHarmonyTerminator(
        in text: String,
        after start: String.Index
    ) -> Range<String.Index>? {
        ["<|call|>", "<|end|>", "<|return|>"]
            .compactMap { text.range(of: $0, range: start..<text.endIndex) }
            .min { $0.lowerBound < $1.lowerBound }
    }
}
