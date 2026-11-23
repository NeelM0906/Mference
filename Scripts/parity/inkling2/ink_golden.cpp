// ink-golden: Inkling 2 Mini goldens from stock upstream llama.cpp (the Inkling
// program's base reference, tag b12641 or later), for the native runner's
// agreement gates (G6). Built by Scripts/parity/inkling2/llama/Makefile.
//
//   ink-golden tokenize --model <gguf> --in <texts.json>   --out <file.json>
//   ink-golden chat     --model <gguf> --in <chat.json>    --out <file.json>
//   ink-golden generate --model <gguf> --in <prompts.json> --out <dir> [--ngl N] [--threads N]
//                       [--n-ctx N] [--ubatch N] [--top K]
//
// tokenize: every text through llama_tokenize twice, parse_special false (plain
//   text, as user input is tokenized) and true (special-token markup parsed),
//   no BOS added (the model adds none); pieces and the detokenized round trip.
// chat: the model's own chat template rendered by llama.cpp's Jinja renderer
//   (common_chat_templates_apply, the path llama-server takes), with
//   chat_template_kwargs (reasoning_effort) and tools; the rendered string and
//   its tokens (the whole string, parse_special true, as llama-server does).
//   Vocabulary only: no weights are loaded.
// generate: for each prompt (raw text, chat messages, or "ids": token ids used
//   as given, for teacher forcing) prefill in ubatches,
//   then greedy decode one token per step (argmax of the logits, ties to the
//   lower id). Writes the tokens and text; with "logprobs" the top-K
//   log-probabilities (log-softmax over the 200,058 real tokens) at every
//   prompt and generated position; with "trace" the hidden state after each of
//   the 48 layer visits (graph tensor l_out-<slot>) at every prompt position,
//   the eight routed expert ids and their weights per visit, and the final
//   normed hidden state, captured with the scheduler's eval callback during
//   the prefill.
// Everything is described by JSON next to the binary arrays (little-endian F32,
// row-major [positions x 2304]). One process runs a whole input file: it is one
// model command under the program's machine lock.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "chat.h"
#include "common.h"
#include "json.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using njson = nlohmann::ordered_json;

// ---------------------------------------------------------------- utilities

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "ink-golden: cannot read %s\n", path.c_str()); exit(2); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static void write_file(const std::string & path, const std::string & text) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "ink-golden: cannot write %s\n", path.c_str()); exit(1); }
    f << text;
    f.close();
    if (!f) { fprintf(stderr, "ink-golden: short write to %s\n", path.c_str()); exit(1); }
}

static void write_f32(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "ink-golden: cannot write %s\n", path.c_str()); exit(1); }
    f.write(reinterpret_cast<const char *>(v.data()), (std::streamsize) (v.size() * sizeof(float)));
    f.close();
    if (!f) { fprintf(stderr, "ink-golden: short write to %s\n", path.c_str()); exit(1); }
}

static void mkdirs(const std::string & path) {
    std::string cur;
    std::stringstream ss(path); std::string part;
    if (!path.empty() && path[0] == '/') cur = "/";
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        cur += part + "/";
        mkdir(cur.c_str(), 0755);
    }
}

static std::string piece(const llama_vocab * vocab, llama_token t) {
    char buf[512];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::string s((size_t) -n, '\0');
        n = llama_token_to_piece(vocab, t, s.data(), (int32_t) s.size(), 0, true);
        return s.substr(0, (size_t) std::max(n, 0));
    }
    return std::string(buf, (size_t) n);
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool parse_special) {
    std::vector<llama_token> out(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                           /*add_special*/ true, parse_special);
    if (n < 0) {
        out.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), true, parse_special);
    }
    if (n < 0) { fprintf(stderr, "ink-golden: tokenize failed\n"); exit(1); }
    out.resize((size_t) n);
    return out;
}

static std::string detok(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    std::string s;
    for (auto t : ids) s += piece(vocab, t);
    return s;
}

static std::string arg(int argc, char ** argv, const char * name, const char * def = nullptr) {
    for (int i = 2; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    if (def == nullptr) { fprintf(stderr, "ink-golden: missing %s\n", name); exit(2); }
    return def;
}

// ngl 0 means CPU only: no GPU device at all (so no op is offloaded to Metal)
// and no extra buffer types (CPU_REPACK would copy every Q8_0 weight it can
// repack into heap memory instead of reading the mapped file).
static ggml_backend_dev_t g_no_devices[1] = { nullptr };

static llama_model * load_model(const std::string & path, bool vocab_only, int ngl) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only   = vocab_only;
    mp.n_gpu_layers = ngl;
    if (ngl == 0) {
        mp.devices         = g_no_devices;
        mp.use_extra_bufts = false;
    }
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) { fprintf(stderr, "ink-golden: cannot load %s\n", path.c_str()); exit(1); }
    return model;
}

// ---------------------------------------------------------------- chat

struct rendered_chat {
    std::string text;
    std::vector<llama_token> ids;
};

static rendered_chat render_chat(const llama_model * model, const common_chat_templates * tmpls, const njson & c) {
    common_chat_templates_inputs in;
    in.messages  = common_chat_msgs_parse_oaicompat(common_json::parse(c.at("messages").dump()));
    in.use_jinja = true;
    in.add_generation_prompt = c.value("add_generation_prompt", true);
    if (c.contains("tools")) {
        in.tools = common_chat_tools_parse_oaicompat(common_json::parse(c.at("tools").dump()));
    }
    if (c.contains("reasoning_effort")) {
        in.chat_template_kwargs["reasoning_effort"] = c.at("reasoning_effort").dump();
    }
    common_chat_params p = common_chat_templates_apply(tmpls, in);
    rendered_chat r;
    r.text = p.prompt.str();
    r.ids  = tokenize(llama_model_get_vocab(model), r.text, /*parse_special*/ true);
    return r;
}

static njson ids_json(const std::vector<llama_token> & ids) {
    njson a = njson::array();
    for (auto t : ids) a.push_back(t);
    return a;
}

static njson pieces_json(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    njson a = njson::array();
    for (auto t : ids) a.push_back(piece(vocab, t));
    return a;
}

// a token's piece can be part of a multi-byte UTF-8 character: the readable
// strings get U+FFFD for such bytes (see dump_out), so the exact bytes of every
// piece are also recorded in hex
static njson pieces_hex_json(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    static const char * hx = "0123456789abcdef";
    njson a = njson::array();
    for (auto t : ids) {
        const std::string p = piece(vocab, t);
        std::string h;
        for (unsigned char c : p) { h += hx[c >> 4]; h += hx[c & 15]; }
        a.push_back(h);
    }
    return a;
}

static std::string dump_out(const njson & j, int indent = -1) {
    return j.dump(indent, ' ', false, njson::error_handler_t::replace);
}

// ---------------------------------------------------------------- trace (eval callback)

static const int N_EMBD = 2304;

struct trace_state {
    bool active = false;
    int  n_tokens = 0;
    std::map<int, std::vector<float>>   l_out;   // slot -> [n_tokens x n_embd]
    std::map<int, std::vector<int32_t>> topk;    // slot -> [n_tokens x 8]
    std::map<int, std::vector<float>>   weights; // slot -> [n_tokens x 8] (after normalization and scale)
    std::vector<float> result_norm;
};

static trace_state g_trace;

// exactly "<base>-<digits>" (not the "... (reshaped)" / "... (view)" copies)
static bool want(const char * name, std::string & base, int & il) {
    const char * dash = strrchr(name, '-');
    if (!dash || !dash[1]) return false;
    for (const char * p = dash + 1; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    base.assign(name, (size_t) (dash - name));
    il = atoi(dash + 1);
    return base == "l_out" || base == "ffn_moe_topk" || base == "ffn_moe_weights_scaled";
}

// copy a tensor of up to 2 dims (rows may be strided) as rows of ne0 elements
template <typename T>
static std::vector<T> fetch_2d(const ggml_tensor * t) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1] * t->ne[2] * t->ne[3];
    std::vector<T> out((size_t) (ne0 * ne1));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(T));
        return out;
    }
    for (int64_t r = 0; r < ne1; ++r) {
        const int64_t i1 = r % t->ne[1], i2 = (r / t->ne[1]) % t->ne[2], i3 = r / (t->ne[1] * t->ne[2]);
        const size_t off = (size_t) (i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
        ggml_backend_tensor_get(t, out.data() + r * ne0, off, (size_t) ne0 * sizeof(T));
    }
    return out;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    if (!g_trace.active) return false;
    std::string base; int il = -1;
    const bool w = want(t->name, base, il) || strcmp(t->name, "result_norm") == 0;
    if (ask) return w;
    if (!w) return true;
    if (strcmp(t->name, "result_norm") == 0) {
        g_trace.result_norm = fetch_2d<float>(t);
    } else if (base == "l_out") {
        if (t->type != GGML_TYPE_F32 || t->ne[0] != N_EMBD) { fprintf(stderr, "ink-golden: unexpected %s\n", t->name); exit(1); }
        g_trace.l_out[il] = fetch_2d<float>(t);
    } else if (base == "ffn_moe_topk") {
        if (t->type != GGML_TYPE_I32) { fprintf(stderr, "ink-golden: unexpected %s type\n", t->name); exit(1); }
        g_trace.topk[il] = fetch_2d<int32_t>(t);
    } else if (base == "ffn_moe_weights_scaled") {
        g_trace.weights[il] = fetch_2d<float>(t);
    }
    return true;
}

// slot of the v-th layer visit (layer_phys in src/models/inkling2-loop.cpp, the
// order of graph::graph there): prelude, core passes 1..P, coda
static std::vector<std::pair<int, int>> visit_slots(int n_phys, int start, int end, int passes) {
    std::vector<std::pair<int, int>> v; // (slot, physical layer)
    for (int il = 0; il < start; ++il) v.push_back({il, il});
    for (int p = 0; p < passes; ++p) {
        for (int ip = start; ip < end; ++ip) {
            const int il = p == 0 ? ip : n_phys + (p - 1) * (end - start) + (ip - start);
            v.push_back({il, ip});
        }
    }
    for (int il = end; il < n_phys; ++il) v.push_back({il, il});
    return v;
}

// ---------------------------------------------------------------- generate

struct topk_entry { int id; float lp; };

static std::vector<topk_entry> top_logprobs(const float * logits, int n_vocab, int k, float * lse_out) {
    float mx = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) mx = std::max(mx, logits[i]);
    double s = 0.0;
    for (int i = 0; i < n_vocab; ++i) s += std::exp((double) logits[i] - mx);
    const float lse = mx + (float) std::log(s);
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    std::vector<topk_entry> out;
    for (int i = 0; i < k; ++i) out.push_back({idx[i], logits[idx[i]] - lse});
    if (lse_out) *lse_out = lse;
    return out;
}

static int argmax(const float * logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
    return best;
}

static int cmd_generate(int argc, char ** argv) {
    const std::string model_path = arg(argc, argv, "--model");
    const std::string out_dir    = arg(argc, argv, "--out");
    const njson prompts = njson::parse(read_file(arg(argc, argv, "--in")));
    const int ngl     = atoi(arg(argc, argv, "--ngl", "99").c_str());
    const int threads = atoi(arg(argc, argv, "--threads", "8").c_str());
    const int n_ctx   = atoi(arg(argc, argv, "--n-ctx", "4096").c_str());
    const int ubatch  = atoi(arg(argc, argv, "--ubatch", "512").c_str());
    const int top_k   = atoi(arg(argc, argv, "--top", "20").c_str());
    mkdirs(out_dir);

    llama_model * model = load_model(model_path, false, ngl);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto tmpls = common_chat_templates_init(model, "");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = ubatch;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.cb_eval = eval_cb; cp.cb_eval_user_data = nullptr;
    cp.no_perf = false;
    cp.op_offload = ngl != 0; // CPU only: no host-weight op goes to the GPU
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ink-golden: cannot create context\n"); return 1; }

    char desc[256]; llama_model_desc(model, desc, sizeof(desc));
    njson index = njson::object();
    index["format"] = "ink-golden-v1";
    index["model"] = model_path;
    index["model_desc"] = desc;
    index["llama_build"] = llama_print_system_info();
    index["n_gpu_layers"] = ngl; index["threads"] = threads; index["n_ctx"] = n_ctx;
    index["n_batch"] = n_ctx; index["n_ubatch"] = ubatch; index["n_vocab"] = n_vocab;
    index["cpu_only_devices"] = ngl == 0; index["use_extra_bufts"] = ngl != 0; index["op_offload"] = cp.op_offload;
    index["flash_attn_type"] = (int) cp.flash_attn_type; index["type_k"] = ggml_type_name(cp.type_k);
    index["type_v"] = ggml_type_name(cp.type_v);
    index["prompts"] = njson::array();

    for (const auto & pr : prompts) {
        const std::string id = pr.at("id");
        const std::string kind = pr.value("kind", "raw");
        const int n_gen = pr.value("n_gen", 32);
        const bool want_lp = pr.value("logprobs", false);
        // "logprobs_last": N records only the last N prompt positions (and every
        // generated one), and requests logits only there: a 2,600-token prompt
        // with logits everywhere would need a 2 GB output buffer
        const int lp_last = pr.value("logprobs_last", 0);
        const bool want_trace = pr.value("trace", false);
        std::vector<llama_token> ids;
        std::string prompt_text;
        if (kind == "chat") {
            rendered_chat r = render_chat(model, tmpls.get(), pr);
            prompt_text = r.text; ids = r.ids;
        } else if (kind == "ids") {
            // token ids given directly (teacher forcing: another run's prompt
            // ids followed by its generated ids), never re-tokenized
            for (const auto & t : pr.at("ids")) ids.push_back((llama_token) t.get<int>());
            prompt_text = detok(vocab, ids);
        } else {
            prompt_text = pr.at("text");
            ids = tokenize(vocab, prompt_text, pr.value("parse_special", false));
        }
        const int n_prompt = (int) ids.size();
        if (n_prompt + n_gen > n_ctx) { fprintf(stderr, "ink-golden: %s too long\n", id.c_str()); return 1; }
        // the trace keeps one ubatch's tensors: the whole prompt must be one ubatch
        if (want_trace && n_prompt > ubatch) {
            fprintf(stderr, "ink-golden: %s: a trace prompt must fit one ubatch (%d > %d)\n", id.c_str(), n_prompt, ubatch);
            return 1;
        }
        fprintf(stderr, "ink-golden: %s: %d prompt tokens, %d to generate%s%s\n", id.c_str(), n_prompt, n_gen,
                want_lp ? ", logprobs" : "", want_trace ? ", trace" : "");

        llama_memory_clear(llama_get_memory(ctx), true);
        const std::string pdir = out_dir + "/" + id;
        mkdirs(pdir);

        njson lp_rows = njson::array();
        auto record_lp = [&](int pos, const float * logits, int next_token) {
            float lse = 0;
            auto top = top_logprobs(logits, n_vocab, top_k, &lse);
            njson row = njson::object();
            row["pos"] = pos;
            row["next"] = next_token;
            row["next_logprob"] = next_token >= 0 ? (double) (logits[next_token] - lse) : 0.0;
            njson t = njson::array();
            for (auto & e : top) t.push_back(njson::array({e.id, e.lp}));
            row["top"] = t;
            lp_rows.push_back(row);
        };

        // prefill: every position outputs logits when logprobs are wanted, else the last only
        g_trace = trace_state();
        g_trace.active = want_trace;
        g_trace.n_tokens = n_prompt;
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        batch.n_tokens = n_prompt;
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[i] = ids[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            const bool lp_here = want_lp && (lp_last <= 0 || i >= n_prompt - 1 - lp_last);
            batch.logits[i] = (lp_here || i == n_prompt - 1) ? 1 : 0;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "ink-golden: prefill failed for %s\n", id.c_str()); return 1; }
        const auto t1 = std::chrono::steady_clock::now();
        g_trace.active = false;
        if (want_lp) {
            const int first = lp_last > 0 ? std::max(0, n_prompt - 1 - lp_last) : 0;
            for (int i = first; i + 1 < n_prompt; ++i) record_lp(i, llama_get_logits_ith(ctx, i), ids[i + 1]);
        }
        std::vector<llama_token> gen;
        std::vector<float> last_logits(llama_get_logits_ith(ctx, n_prompt - 1), llama_get_logits_ith(ctx, n_prompt - 1) + n_vocab);
        llama_batch_free(batch);

        bool eog = false;
        llama_batch one = llama_batch_init(1, 0, 1);
        for (int s = 0; s < n_gen; ++s) {
            const int next = argmax(last_logits.data(), n_vocab);
            if (want_lp) record_lp(n_prompt - 1 + s, last_logits.data(), next);
            gen.push_back(next);
            if (llama_vocab_is_eog(vocab, next)) { eog = true; break; }
            if (s + 1 == n_gen) break;
            one.n_tokens = 1; one.token[0] = next; one.pos[0] = n_prompt + s; one.n_seq_id[0] = 1;
            one.seq_id[0][0] = 0; one.logits[0] = 1;
            if (llama_decode(ctx, one) != 0) { fprintf(stderr, "ink-golden: decode failed for %s\n", id.c_str()); return 1; }
            const float * l = llama_get_logits_ith(ctx, 0);
            last_logits.assign(l, l + n_vocab);
        }
        llama_batch_free(one);
        const auto t2 = std::chrono::steady_clock::now();

        njson rec = njson::object();
        rec["id"] = id;
        rec["kind"] = kind;
        rec["prompt_text"] = prompt_text;
        rec["prompt_ids"] = ids_json(ids);
        rec["generated_ids"] = ids_json(gen);
        rec["generated_text"] = detok(vocab, gen);
        rec["generated_pieces"] = pieces_json(vocab, gen);
        rec["stop"] = eog ? "eog" : "n_gen";
        rec["prefill_s"] = std::chrono::duration<double>(t1 - t0).count();
        rec["decode_s"] = std::chrono::duration<double>(t2 - t1).count();
        if (want_lp) {
            rec["logprobs_file"] = "logprobs.json";
            rec["logprobs_first_pos"] = lp_last > 0 ? std::max(0, n_prompt - 1 - lp_last) : 0;
            write_file(pdir + "/logprobs.json", dump_out(lp_rows));
        }
        if (want_trace) {
            const njson & hp = pr.at("loop"); // {n_phys, start, end, passes}
            auto visits = visit_slots(hp.at("n_phys"), hp.at("start"), hp.at("end"), hp.at("passes"));
            njson tv = njson::array();
            std::vector<float> states;
            for (size_t v = 0; v < visits.size(); ++v) {
                const int slot = visits[v].first;
                if (!g_trace.l_out.count(slot) || !g_trace.topk.count(slot)) {
                    fprintf(stderr, "ink-golden: no trace for slot %d\n", slot); return 1;
                }
                const auto & h = g_trace.l_out[slot];
                if ((int) h.size() != n_prompt * N_EMBD) { fprintf(stderr, "ink-golden: slot %d size %zu\n", slot, h.size()); return 1; }
                states.insert(states.end(), h.begin(), h.end());
                njson e = njson::object();
                e["visit"] = (int) v; e["slot"] = slot; e["layer"] = visits[v].second;
                njson ex = njson::array(), wx = njson::array();
                const auto & k = g_trace.topk[slot];
                const int kk = (int) (k.size() / n_prompt);
                for (int t = 0; t < n_prompt; ++t) {
                    njson row = njson::array(), wrow = njson::array();
                    for (int j = 0; j < kk; ++j) row.push_back(k[t * kk + j]);
                    if (g_trace.weights.count(slot)) {
                        const auto & w = g_trace.weights[slot];
                        const int wk = (int) (w.size() / n_prompt);
                        for (int j = 0; j < wk; ++j) wrow.push_back(w[t * wk + j]);
                    }
                    ex.push_back(row); wx.push_back(wrow);
                }
                e["experts"] = ex; e["expert_weights"] = wx;
                tv.push_back(e);
            }
            write_f32(pdir + "/hidden_after_visit.f32", states);
            write_f32(pdir + "/result_norm.f32", g_trace.result_norm);
            njson tr = njson::object();
            tr["hidden_file"] = "hidden_after_visit.f32";
            tr["hidden_shape"] = njson::array({(int) visits.size(), n_prompt, N_EMBD});
            tr["hidden_tensor"] = "l_out-<slot> (after the layer's MoE residual add)";
            tr["result_norm_file"] = "result_norm.f32";
            // only the positions that produced logits: every position with logprobs,
            // the last N + 1 with logprobs_last N, else the last one
            tr["result_norm_shape"] = njson::array({(int) (g_trace.result_norm.size() / N_EMBD), N_EMBD});
            tr["visits"] = tv;
            write_file(pdir + "/trace.json", dump_out(tr));
            rec["trace_file"] = "trace.json";
        }
        write_file(pdir + "/golden.json", dump_out(rec, 1));
        njson brief = njson::object();
        brief["id"] = id; brief["n_prompt"] = n_prompt; brief["n_generated"] = (int) gen.size(); brief["stop"] = rec["stop"];
        index["prompts"].push_back(brief);
        fprintf(stderr, "ink-golden: %s: %zu generated (%s) in %.2f s prefill, %.2f s decode\n", id.c_str(), gen.size(),
                eog ? "eog" : "n_gen", (double) rec["prefill_s"], (double) rec["decode_s"]);
    }
    write_file(out_dir + "/index.json", dump_out(index, 1));
    llama_perf_context_print(ctx);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

// ---------------------------------------------------------------- tokenize, chat

static int cmd_tokenize(int argc, char ** argv) {
    llama_model * model = load_model(arg(argc, argv, "--model"), true, 0);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const njson texts = njson::parse(read_file(arg(argc, argv, "--in")));
    njson out = njson::array();
    for (const auto & t : texts) {
        const std::string text = t.at("text");
        njson r = njson::object();
        r["id"] = t.at("id");
        r["text"] = text;
        for (bool special : {false, true}) {
            auto ids = tokenize(vocab, text, special);
            njson s = njson::object();
            s["ids"] = ids_json(ids);
            s["pieces"] = pieces_json(vocab, ids);
            s["pieces_hex"] = pieces_hex_json(vocab, ids);
            s["roundtrip_equal"] = detok(vocab, ids) == text;
            r[special ? "parse_special" : "plain"] = s;
        }
        out.push_back(r);
    }
    write_file(arg(argc, argv, "--out"), dump_out(out, 1));
    llama_model_free(model);
    return 0;
}

static int cmd_chat(int argc, char ** argv) {
    llama_model * model = load_model(arg(argc, argv, "--model"), true, 0);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, "");
    const njson cases = njson::parse(read_file(arg(argc, argv, "--in")));
    njson out = njson::array();
    for (const auto & c : cases) {
        njson r = njson::object();
        r["id"] = c.at("id");
        try {
            rendered_chat rc = render_chat(model, tmpls.get(), c);
            r["rendered"] = rc.text;
            r["ids"] = ids_json(rc.ids);
            r["pieces"] = pieces_json(vocab, rc.ids);
        } catch (const std::exception & e) {
            r["error"] = e.what();
        }
        out.push_back(r);
    }
    write_file(arg(argc, argv, "--out"), dump_out(out, 1));
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: ink-golden tokenize|chat|generate --model <gguf> --in <json> --out <path> [options]\n");
        return 2;
    }
    llama_backend_init();
    const std::string cmd = argv[1];
    int rc = 2;
    if (cmd == "tokenize") rc = cmd_tokenize(argc, argv);
    else if (cmd == "chat") rc = cmd_chat(argc, argv);
    else if (cmd == "generate") rc = cmd_generate(argc, argv);
    else fprintf(stderr, "ink-golden: unknown command %s\n", cmd.c_str());
    llama_backend_free();
    return rc;
}
