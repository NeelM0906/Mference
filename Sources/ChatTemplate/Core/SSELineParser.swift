import Foundation

public enum SSEEvent: Equatable, Sendable {
    case data(String)
    case done
}

/// Parses server-sent-event lines from an OpenAI-compatible
/// `/v1/chat/completions` stream.
public enum SSELineParser {
    /// Returns nil for lines that carry no event (blank lines, comments,
    /// non-data fields).
    public static func parse(line: String) -> SSEEvent? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("data:") else { return nil }
        let payload = trimmed.dropFirst("data:".count)
            .trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" { return .done }
        guard !payload.isEmpty else { return nil }
        return .data(payload)
    }

    /// Extracts `choices[0].delta.content` from a `chat.completion.chunk`
    /// JSON payload. Returns nil for chunks without text (role headers,
    /// finish_reason, usage).
    public static func contentDelta(fromChunkJSON json: String) -> String? {
        guard let data = json.data(using: .utf8),
              let chunk = try? JSONDecoder().decode(ChunkEnvelope.self, from: data),
              let content = chunk.choices.first?.delta?.content,
              !content.isEmpty
        else { return nil }
        return content
    }

    private struct ChunkEnvelope: Decodable {
        struct Choice: Decodable {
            struct Delta: Decodable {
                let content: String?
            }

            let delta: Delta?
        }

        let choices: [Choice]
    }
}
