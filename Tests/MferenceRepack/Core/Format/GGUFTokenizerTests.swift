import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct GGUFTokenizerTests {
    private func metadata(types: [Int32] = [1, 1, 1, 6, 5, 3, 3, 4],
                          merges: [String] = ["Ġ a", "Ġa b"],
                          pre: String = "qwen35") -> [String: GGUF.Value] {
        let tokens = ["Ġ", "a", "b", "Ā", "[PAD0]", "<|endoftext|>", "<|im_end|>", "<think>"]
        return [
            "general.architecture": .string("qwen4exp"),
            "qwen4exp.context_length": .u32(262_144),
            "tokenizer.ggml.model": .string("gpt2"),
            "tokenizer.ggml.pre": .string(pre),
            "tokenizer.ggml.tokens": .array(tokens.map { .string($0) }),
            "tokenizer.ggml.token_type": .array(types.map { .i32($0) }),
            "tokenizer.ggml.merges": .array(merges.map { .string($0) }),
            "tokenizer.ggml.bos_token_id": .u32(5),
            "tokenizer.ggml.eos_token_id": .u32(6),
            "tokenizer.ggml.padding_token_id": .u32(5),
            "tokenizer.ggml.add_bos_token": .bool(false),
            "tokenizer.chat_template": .string("{{ messages }}"),
        ]
    }

    @Test func synthesizesBPETokenizerFromMetadata() throws {
        let s = try GGUFTokenizer.synthesize(metadata: metadata(), path: "s1.gguf")
        let t = try #require(try JSONSerialization.jsonObject(with: s.tokenizerJSON) as? [String: Any])
        #expect(t["normalizer"] is NSNull)
        let model = try #require(t["model"] as? [String: Any])
        #expect(model["type"] as? String == "BPE")
        let vocab = try #require(model["vocab"] as? [String: Int])
        #expect(vocab == ["Ġ": 0, "a": 1, "b": 2, "Ā": 3, "[PAD0]": 4])
        #expect(model["merges"] as? [[String]] == [["Ġ", "a"], ["Ġa", "b"]])
        let added = try #require(t["added_tokens"] as? [[String: Any]])
        #expect(added.map { $0["id"] as? Int } == [5, 6, 7])
        #expect(added.map { $0["special"] as? Bool } == [true, true, false])
        let pre = try #require(t["pre_tokenizer"] as? [String: Any])
        let split = try #require((pre["pretokenizers"] as? [[String: Any]])?.first)
        #expect((split["pattern"] as? [String: String])?["Regex"] == GGUFTokenizer.preTokenizerRegex["qwen35"])

        let c = try #require(try JSONSerialization.jsonObject(with: s.tokenizerConfigJSON) as? [String: Any])
        #expect(c["eos_token"] as? String == "<|im_end|>")
        #expect(c["pad_token"] as? String == "<|endoftext|>")
        #expect(c["add_bos_token"] as? Bool == false)
        #expect(s.chatTemplate == "{{ messages }}")
    }

    @Test func qwen35RegexIsTheVendorForm() {
        // The form llama.cpp b1b770d9 quotes as the original tokenizer.json regex.
        #expect(GGUFTokenizer.preTokenizerRegex["qwen35"]
                == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
    }

    @Test func failsClosedOnUnknownPreTokenizerOrBadMerges() {
        #expect(throws: RepackError.self) {
            try GGUFTokenizer.synthesize(metadata: metadata(pre: "llama3"), path: "s1.gguf")
        }
        #expect(throws: RepackError.self) {
            try GGUFTokenizer.synthesize(metadata: metadata(merges: ["Ġ a b"]), path: "s1.gguf")
        }
        #expect(throws: RepackError.self) {
            try GGUFTokenizer.synthesize(metadata: metadata(types: [1, 1, 1, 6, 5, 3, 3, 2]), path: "s1.gguf")
        }
    }
}
