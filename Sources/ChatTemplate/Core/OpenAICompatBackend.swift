import Foundation

public struct OpenAICompatError: Error, LocalizedError, Equatable {
    public let status: Int?
    public let message: String

    public init(status: Int? = nil, message: String) {
        self.status = status
        self.message = message
    }

    public var errorDescription: String? {
        if let status { return "Server error (\(status)): \(message)" }
        return message
    }
}

/// Streams completions from any OpenAI-compatible server
/// (`POST {base}/chat/completions` with `stream: true`), such as the local
/// MferenceServer at `http://127.0.0.1:8080/v1`.
public struct OpenAICompatBackend: ChatBackend {
    public let baseURL: URL
    public let apiKey: String?

    public init(baseURL: URL, apiKey: String? = nil) {
        self.baseURL = baseURL
        self.apiKey = apiKey
    }

    public func stream(
        messages: [ChatMessage],
        model: String
    ) -> AsyncThrowingStream<String, Error> {
        let request = makeRequest(messages: messages, model: model)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    if (response as? HTTPURLResponse)?.statusCode != 200 {
                        // Drain the (bounded) error body so the server's
                        // message survives into the thrown error.
                        var body = Data()
                        for try await byte in bytes.prefix(16_384) { body.append(byte) }
                        try Self.checkStatus(response, body: body)
                    }
                    for try await line in bytes.lines {
                        guard let event = SSELineParser.parse(line: line) else { continue }
                        switch event {
                        case .done:
                            continuation.finish()
                            return
                        case .data(let json):
                            if let delta = SSELineParser.contentDelta(fromChunkJSON: json) {
                                continuation.yield(delta)
                            } else if let message = Self.errorMessage(fromEventJSON: json) {
                                throw OpenAICompatError(message: message)
                            }
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func listModels() async throws -> [ChatModelInfo] {
        var request = URLRequest(url: baseURL.appending(path: "models"))
        applyHeaders(to: &request)
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.checkStatus(response, body: data)
        struct ModelList: Decodable {
            struct Model: Decodable {
                let id: String
            }

            let data: [Model]
        }
        let list = try JSONDecoder().decode(ModelList.self, from: data)
        return list.data.map { ChatModelInfo(id: $0.id, label: $0.id) }
    }

    private func makeRequest(messages: [ChatMessage], model: String) -> URLRequest {
        var request = URLRequest(url: baseURL.appending(path: "chat/completions"))
        request.httpMethod = "POST"
        applyHeaders(to: &request)
        let body: [String: Any] = [
            "model": model,
            "stream": true,
            "messages": messages.map { ["role": $0.role.rawValue, "content": $0.content] },
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        return request
    }

    private func applyHeaders(to request: inout URLRequest) {
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
    }

    private static func checkStatus(_ response: URLResponse, body: Data? = nil) throws {
        guard let http = response as? HTTPURLResponse else {
            throw OpenAICompatError(message: "Not an HTTP response")
        }
        guard http.statusCode == 200 else {
            let message = body.flatMap { Self.errorMessage(fromBody: $0) }
                ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw OpenAICompatError(status: http.statusCode, message: message)
        }
    }

    /// OpenAI-style error bodies: {"error": {"message": "..."}}
    static func errorMessage(fromBody data: Data) -> String? {
        struct ErrorEnvelope: Decodable {
            struct Body: Decodable {
                let message: String?
            }

            let error: Body?
        }
        let envelope = try? JSONDecoder().decode(ErrorEnvelope.self, from: data)
        return envelope?.error?.message
    }

    /// Mid-stream error events use the same envelope inside a data: line.
    static func errorMessage(fromEventJSON json: String) -> String? {
        guard let data = json.data(using: .utf8) else { return nil }
        return errorMessage(fromBody: data)
    }
}
