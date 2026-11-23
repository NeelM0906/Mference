#!/usr/bin/env python3
"""Generates Tests/Mference/Core/Tokenization/Inkling2ToolArgumentsGoldenTests.swift
from K0's chat_cases_tool_args.json (inputs) and chat_goldens_tool_args.json (llama.cpp renders).
Usage: gen_tool_args_golden_test.py CASES.json GOLDENS.json OUT.swift"""
import hashlib, json, sys

cases_path, goldens_path, out_path = sys.argv[1:4]
cases_bytes = open(cases_path, 'rb').read()
goldens_bytes = open(goldens_path, 'rb').read()
cases = json.loads(cases_bytes)
goldens = {g['id']: g for g in json.loads(goldens_bytes)}
assert [c['id'] for c in cases] == sorted(goldens), (sorted(goldens), [c['id'] for c in cases])


def lit(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == '\\': out.append('\\\\')
        elif ch == '"': out.append('\\"')
        elif ch == '\n': out.append('\\n')
        elif ch == '\r': out.append('\\r')
        elif ch == '\t': out.append('\\t')
        elif o < 0x20 or o == 0x7f: out.append('\\u{%x}' % o)
        else: out.append(ch)
    return '"' + ''.join(out) + '"'


tool = None
rows = []
for c in cases:
    msgs = c['messages']
    assert [m['role'] for m in msgs][:2] == ['user', 'assistant'], c['id']
    assert msgs[0]['content'] == 'Look this up in my records.', c['id']
    calls = msgs[1]['tool_calls']
    results = msgs[2:]
    assert [r['tool_call_id'] for r in results] == [k['id'] for k in calls], c['id']
    assert all(r['content'] == 'ok' and r['role'] == 'tool' for r in results), c['id']
    assert all(k['function']['name'] == 'query_records' for k in calls), c['id']
    assert len(c['tools']) == 1
    t = c['tools'][0]['function']
    assert tool in (None, t); tool = t
    g = goldens[c['id']]
    calls_swift = ', '.join('(%s, %s)' % (lit(k['id']), lit(k['function']['arguments'])) for k in calls)
    rows.append('        .init(id: %s,\n              calls: [%s],\n              rendered: %s,\n              ids: %s),'
                % (lit(c['id']), calls_swift, lit(g['rendered']), json.dumps(g['ids'])))

assert tool['parameters'] == {'type': 'object'}
cases_sha = hashlib.sha256(cases_bytes).hexdigest()
goldens_sha = hashlib.sha256(goldens_bytes).hexdigest()

swift = '''import Foundation
import Testing
@testable import Mference

/// The Inkling 2 render of tool-call histories against K0's tool-argument
/// goldens: llama.cpp b12641 rendering conversations whose call arguments are
/// JSON strings (`chat_cases_tool_args.json`, SHA-256 `%(cases_sha)s`;
/// renders `chat_goldens_tool_args.json`, SHA-256 `%(goldens_sha)s`, both in
/// `Tests/Mference/Fixtures/Inkling2/`). The cases and renders below are
/// generated from those files, not retyped. Each case: a user message, the
/// assistant's call(s) to `query_records`, an "ok" result per call.
///
/// `t07` holds an integer above Int64, where the references disagree:
/// llama.cpp converts every JSON integer with `get<int64_t>()` and wraps it,
/// Python keeps the digits. Mference refuses it, as before (the server answers
/// 400, "historical tool arguments cannot be represented exactly").
///
/// With `MFERENCE_INKLING2_TOKENIZER_DIR` (the real vocabulary) the token ids
/// are compared with llama.cpp's too; skipped otherwise.
@Suite("Inkling 2 tool-call argument goldens")
struct Inkling2ToolArgumentsGoldenTests {
    private typealias Message = MFTokenizer.Message

    struct Case {
        let id: String
        let calls: [(id: String, arguments: String)]
        let rendered: String
        let ids: [Int32]
    }

    static let tool = MFTokenizer.FunctionDefinition(name: %(tool_name)s, description: %(tool_desc)s,
                                                     parameters: .object(["type": .string("object")]))

    static let cases: [Case] = [
%(rows)s
    ]

    /// Cases whose arguments a template value carries exactly.
    static var exact: [Case] { cases.filter { $0.id != "t07-int-beyond-int64" } }

    /// The case's conversation, each call's arguments given as `texts` (by
    /// default the case's own raw strings).
    static func messages(_ c: Case, texts: [String]? = nil) throws -> [MFTokenizer.Message] {
        let texts = texts ?? c.calls.map(\\.arguments)
        let calls = try zip(c.calls, texts).map { call, text in
            MFTokenizer.HistoricalToolCall(
                id: call.id, name: "query_records",
                arguments: try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)),
                argumentsText: text)
        }
        return [Message(role: .user, content: "Look this up in my records."),
                Message(role: .assistant, content: "", toolCalls: calls)]
            + c.calls.map { Message(role: .tool, content: "ok", toolCallID: $0.id) }
    }

    @Test func rendersMatchTheReference() async throws {
        let tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
        for c in Self.exact {
            let rendered = try tok.renderInkling2Chat(messages: try Self.messages(c), tools: [Self.tool])
            #expect(rendered == c.rendered, "\\(c.id)")
        }
    }

    /// The round trip prefix reuse needs: a call the model writes as the
    /// reference renders it goes through the tool-call parser, back out as the
    /// arguments the server returns, and into the next request's history,
    /// which renders the same text again.
    @Test func aReferenceCallReRendersToItself() async throws {
        let tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
        let invoke = "<|content_invoke_tool_json|>", end = "<|end_message|>"
        for c in Self.exact {
            let bodies = c.rendered.components(separatedBy: invoke).dropFirst()
                .map { String($0.components(separatedBy: end)[0]) }
            #expect(bodies.count == c.calls.count, "\\(c.id)")
            let texts = try bodies.map { body in
                try Inkling2ToolCallParser().parse(body, headerName: "query_records",
                                                   allowedTools: ["query_records"], id: "call").argumentsJSON
            }
            let again = try tok.renderInkling2Chat(messages: try Self.messages(c, texts: texts), tools: [Self.tool])
            #expect(again == c.rendered, "\\(c.id)")
        }
    }

    /// `t07`: the reference wraps the integer; Mference refuses to render it.
    @Test func anIntegerBeyondInt64IsRefused() async throws {
        let tok = try await MFTokenizer.load(from: Inkling2ChatTests.fixtureFolder())
        let c = try #require(Self.cases.first { $0.id == "t07-int-beyond-int64" })
        #expect(c.rendered.contains("-6101065172474983726"), "llama.cpp's wrapped value")
        #expect(throws: (any Error).self) {
            _ = try tok.renderInkling2Chat(messages: try Self.messages(c), tools: [Self.tool])
        }
    }

    private static var realTokenizerDir: URL? {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_TOKENIZER_DIR"].map { URL(fileURLWithPath: $0) }
    }

    @Test(.enabled(if: realTokenizerDir != nil, "needs MFERENCE_INKLING2_TOKENIZER_DIR"))
    func tokenIdsMatchTheReference() async throws {
        let tok = try await MFTokenizer.load(from: try #require(Self.realTokenizerDir))
        #expect(tok.dialect == .inkling2)
        for c in Self.exact {
            #expect(tok.encode(c.rendered, addBOS: false) == c.ids, "\\(c.id)")
            #expect(try tok.encodeInkling2Chat(messages: try Self.messages(c), tools: [Self.tool]) == c.ids,
                    "\\(c.id)")
        }
    }
}
''' % dict(cases_sha=cases_sha, goldens_sha=goldens_sha, tool_name=lit(tool['name']),
           tool_desc=lit(tool['description']), rows='\n'.join(rows))
open(out_path, 'w').write(swift)
print('wrote', out_path, len(rows), 'cases')
