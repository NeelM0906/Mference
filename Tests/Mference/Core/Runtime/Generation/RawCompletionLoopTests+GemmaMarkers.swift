import Foundation
import Metal
import Testing
import MferenceValidationSupport

@testable import Mference

/// A byte-fallback run held open when a channel or tool marker arrives must
/// commit under the channel in effect before the marker. The loop flushes the
/// detokenizer ahead of each marker; these tests drive that through the
/// structured decoder the way the server and CLI do (`.tail` through
/// `consumeFlushedText`).
extension RawCompletionLoopTests {
  struct StructuredOutput {
    var content = ""
    var reasoning = ""
    var calls: [ParsedToolCall] = []
  }

  /// Generates `script` token by token after a one-token prompt, then `end`.
  func runGemmaStructured(_ script: [Int32], end: Int32,
                          allowedTools: Set<String> = []) async throws -> StructuredOutput {
    let ctx = try MetalContext()
    let tok = try await MFTokenizer.load()
    // One prompt token: produce call k yields generated token k.
    let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize) { _, call in
      .argmax(call < script.count ? script[call] : end)
    }
    let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
    let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: allowedTools,
                                             idGenerator: { "call_0" })
    var output = StructuredOutput()
    decoder.onReasoning = { output.reasoning += $0 }
    func handle(_ events: [StructuredAssistantEvent]) {
      for event in events {
        switch event {
        case .content(let text): output.content += text
        case .toolCall(let call): output.calls.append(call)
        }
      }
    }
    var failure: Error?
    _ = try await runRawCompletion(producer: producer, tokenizer: tok,
                                   promptIds: [tok.bosID],
                                   config: GenerationConfig(maxNewTokens: 64, temperature: 0),
                                   context: ctx, scratch: scratch,
                                   prefillConfig: .off) { progress in
      do {
        switch progress {
        case .prefill: break
        case .token(_, let id, let delta): handle(try decoder.consume(tokenID: id, delta: delta))
        case .tail(let text): handle(try decoder.consumeFlushedText(text))
        }
      } catch {
        failure = error
      }
    }
    if let failure { throw failure }
    handle(try decoder.finish())
    return output
  }

  func emojiBytes(_ tok: MFTokenizer) throws -> [Int32] {
    try ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"].map { token in
      let id = try #require(tok.tokenizer.convertTokenToId(token))
      return Int32(id)
    }
  }

  @Test func thoughtEndingInByteRunStaysInReasoning() async throws {
    let tok = try await MFTokenizer.load()
    let script: [Int32] = [tok.channelStartID] + tok.encode("thought\nLet me think", addBOS: false)
      + (try emojiBytes(tok)) + [tok.channelEndID] + tok.encode("Hi", addBOS: false)
    let output = try await runGemmaStructured(script, end: tok.endOfTurnID)
    #expect(output.reasoning == "Let me think😀")
    #expect(output.content == "Hi")
  }

  @Test func visibleByteRunBeforeChannelStaysVisible() async throws {
    let tok = try await MFTokenizer.load()
    let script: [Int32] = tok.encode("Hi", addBOS: false) + (try emojiBytes(tok))
      + [tok.channelStartID] + tok.encode("thought\nx", addBOS: false)
      + [tok.channelEndID] + tok.encode(" ok", addBOS: false)
    let output = try await runGemmaStructured(script, end: tok.endOfTurnID)
    #expect(output.content == "Hi😀 ok")
    #expect(output.reasoning == "x")
  }

  @Test func visibleByteRunBeforeToolCallStaysVisible() async throws {
    let tok = try await MFTokenizer.load()
    let script: [Int32] = tok.encode("Sure", addBOS: false) + (try emojiBytes(tok))
      + [tok.toolCallStartID]
      + tok.encode(#"call:get_weather{location:<|"|>Paris , France 😀<|"|>}"#, addBOS: false)
      + [tok.toolCallEndID]
    let output = try await runGemmaStructured(script, end: tok.toolResponseID,
                                              allowedTools: ["get_weather"])
    #expect(output.content == "Sure😀")
    #expect(output.calls.map(\.name) == ["get_weather"])
    #expect(output.calls.first?.arguments == .object(["location": .string("Paris , France 😀")]))
  }
}
