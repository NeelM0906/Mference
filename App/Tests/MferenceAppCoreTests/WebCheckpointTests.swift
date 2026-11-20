import Foundation
import MferenceAppCore
import Testing

/// The C-A2 harness logic, on the scripted engine (no model, no network).
@MainActor
@Suite struct WebCheckpointTests {
    static let call = ToolCall(id: "c1", name: "web_search", arguments: .object(["query": .string("swift release")]),
                               argumentsJSON: #"{"query":"swift release"}"#)

    @Test func searchedAndCitedPasses() async {
        let service = ScriptedEngineService(replies: [
            .init(content: "", reasoning: "Need current facts.", toolCalls: [Self.call]),
            .init(content: "It is on sale through Sunday [1]."),
        ])
        let outcome = await WebCheckpoint.run(question: "Q?", model: .quick, service: service, tools: StubToolbox())
        #expect(outcome.passed, "\(outcome)")
        #expect(outcome.error == nil)
        #expect(outcome.searches == ["oakline duo sale"])
        #expect(outcome.sources.map(\.number) == [1])
        #expect(outcome.sources.first?.url == "https://oakline.example/sale")
        #expect(outcome.finish == "stop")
        #expect(outcome.model == "Quick")
        // The question went through the chat with the web tools offered.
        #expect(service.requests.first?.options.tools.map(\.name) == ["web_search"])
        #expect(service.requests.first?.turns.last?.content == "Q?")
    }

    @Test func answeringWithoutSearchingFails() async {
        let service = ScriptedEngineService(replies: [.init(content: "From memory: 6.0.")])
        let outcome = await WebCheckpoint.run(question: "Q?", model: .deep, service: service, tools: StubToolbox())
        #expect(!outcome.passed)
        #expect(outcome.error == "the model did not search")
        #expect(outcome.answer == "From memory: 6.0.")
    }

    @Test func searchingWithoutCitingFails() async {
        let service = ScriptedEngineService(replies: [
            .init(content: "", toolCalls: [Self.call]),
            .init(content: "No citations here."),
        ])
        let outcome = await WebCheckpoint.run(question: "Q?", model: .quick, service: service, tools: StubToolbox())
        #expect(!outcome.passed)
        #expect(outcome.error == "the answer cites no sources")
        #expect(outcome.searches.count == 1)
    }

    @Test func engineErrorsAreReported() async {
        let service = ScriptedEngineService(replies: [.init(content: "", error: .busy)])
        let outcome = await WebCheckpoint.run(question: "Q?", model: .quick, service: service, tools: StubToolbox())
        #expect(!outcome.passed)
        #expect(outcome.error == EngineServiceError.busy.userMessage)
    }
}
