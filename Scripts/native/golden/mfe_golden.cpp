// mfe-golden: golden capture from pinned llama.cpp (b1b770d9) for the native
// engine's parity harness (docs/NATIVE_PARITY.md).
//
//   mfe-golden --model <gguf> --prompts <set.json> --out <dir> [options]
//
// For every prompt of the set (Tests/Mference/Fixtures/native/prompts/):
//   1. Renders the chat template with llama.cpp's own renderer
//      (common_chat_templates_apply, as llama-server does) and tokenizes the
//      result (add_special and parse_special on, as llama-server does).
//   2. Pass A (the golden): prefills the prompt the way the product's
//      LlamaBackend does (n_batch 2048, n_ubatch 512, F16 KV, flash attention
//      auto, every layer on the chosen backend), then decodes greedily one
//      token per step, keeping the F32 logits row behind every choice.
//      Writes logits.f32 [n_gen x n_vocab] and the tokens.
//      With --teacher <run-dir>, every step feeds that run's generated token
//      instead of its own argmax (teacher forcing, for the noise-floor runs
//      that are compared with the golden position by position); its own
//      argmax is recorded as argmax_tokens.
//   3. Pass B (--mode golden only): a fresh sequence, the prompt in n_ubatch
//      chunks at the same boundaries, then the generated tokens teacher-forced
//      one per step, with an eval callback (ggml_backend_sched_eval_callback)
//      that copies named graph tensors at chosen positions to
//      tensors/<name>.f32 ([positions x cols] F32, row-major, little endian).
//
// Everything is described by <out>/<prompt-id>/golden.json (format
// "mfe-golden-v1"); the Swift loader is Tests/Mference/Core/Native/Harness.
// One process runs the whole set: it is one model command (spec section 10).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "chat.h"
#include "json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <random>

// ---------------------------------------------------------------- utilities

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "mfe-golden: cannot read %s\n", path.c_str()); exit(2); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static void write_file(const std::string & path, const std::string & text) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "mfe-golden: cannot write %s\n", path.c_str()); exit(1); }
    f << text;
    f.close();
    if (!f) { fprintf(stderr, "mfe-golden: short write to %s\n", path.c_str()); exit(1); }
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

static std::string json_str(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char) c;
        }
    }
    return o + "\"";
}

template <typename T>
static std::string json_ints(const std::vector<T> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += std::to_string(v[i]); }
    return s + "]";
}

// [[start, end), ...] from an ascending position list.
static std::string json_ranges(const std::vector<int64_t> & p) {
    std::string s = "[";
    size_t i = 0; bool first = true;
    while (i < p.size()) {
        size_t j = i;
        while (j + 1 < p.size() && p[j + 1] == p[j] + 1) ++j;
        if (!first) s += ",";
        first = false;
        s += "[" + std::to_string(p[i]) + "," + std::to_string(p[j] + 1) + "]";
        i = j + 1;
    }
    return s + "]";
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> v; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, sep)) if (!x.empty()) v.push_back(x);
    return v;
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- capture

// A name to capture: graph tensor base name ("l_out"), an optional
// occurrence filter (#k: the k-th node of that full name in one graph), and
// whether every position or only the selected ones are kept.
struct NameSpec {
    std::string base;
    int occurrence = -1;  // -1: every occurrence
    bool all_rows = false;
    bool state_only = false; // only the first --state-steps decode steps
};

struct CaptureFile {
    std::string rel;      // tensors/<name>.f32
    std::string full;     // name with layer and occurrence
    std::string base;
    int layer = -1;
    int occurrence = 0;
    std::string op;
    int64_t ne[4] = {0, 0, 0, 0};
    int64_t cols = 0;
    std::vector<int64_t> positions;
    FILE * fp = nullptr;
};

struct Capture {
    std::string dir;
    std::vector<NameSpec> specs;
    // current graph
    int64_t chunk_start = 0;
    int64_t chunk_rows = 0;
    std::vector<int64_t> sel_in_chunk;  // chunk-relative rows kept for "sel" names (ascending)
    std::vector<int64_t> out_in_chunk;  // chunk-relative rows whose logits are requested (ascending)
    bool state_step = false;            // this graph is a decode step that keeps state_only names
    std::map<std::string, int> seen;    // full name -> occurrences this graph
    std::map<std::string, CaptureFile> files;
    std::vector<std::string> order;
    std::vector<std::string> skipped;
};

static bool parse_name(const char * full, std::string & base, int & layer) {
    const char * dash = strrchr(full, '-');
    if (dash) {
        char * end = nullptr;
        long il = strtol(dash + 1, &end, 10);
        if (end != dash + 1 && *end == 0) {
            base.assign(full, dash - full);
            layer = (int) il;
            return true;
        }
    }
    base = full;
    layer = -1;
    return true;
}

static const NameSpec * match(const Capture & c, const std::string & base) {
    for (const auto & s : c.specs) if (s.base == base) return &s;
    return nullptr;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user) {
    Capture * c = (Capture *) user;
    std::string base; int layer;
    parse_name(t->name, base, layer);
    const NameSpec * spec = match(*c, base);
    if (!spec || t->type != GGML_TYPE_F32 || (spec->state_only && !c->state_step)) {
        return ask ? false : true;
    }
    const std::string full = t->name;
    if (ask) {
        // a prefill chunk without selected rows keeps nothing for "sel" names:
        // do not split the graph there (each observed node is a sync point)
        return spec->all_rows || c->chunk_rows == 1 || !c->sel_in_chunk.empty();
    }
    const int occ = c->seen[full]++;
    if (spec->occurrence >= 0 && occ != spec->occurrence) return true;

    // token dimension: the highest dimension that is not 1; its extent is the
    // chunk's rows, or the number of output rows (last block after the crop).
    // Dimension 0 is never the token dimension; a tensor whose higher
    // dimensions are all 1 holds a single row (a chunk with one output row).
    int tdim = 1;
    for (int d = 3; d >= 1; --d) { if (t->ne[d] != 1) { tdim = d; break; } }
    const int64_t extent = c->chunk_rows == 1 ? 1 : t->ne[tdim];
    std::vector<int64_t> rows_in_tensor, positions;
    if (c->chunk_rows == 1) {
        // a decode step: the whole tensor is the one row
        tdim = 4;
        rows_in_tensor = {0};
        positions = {c->chunk_start};
    } else if (extent == c->chunk_rows) {
        if (spec->all_rows) {
            for (int64_t r = 0; r < c->chunk_rows; ++r) { rows_in_tensor.push_back(r); positions.push_back(c->chunk_start + r); }
        } else {
            for (int64_t r : c->sel_in_chunk) { rows_in_tensor.push_back(r); positions.push_back(c->chunk_start + r); }
        }
    } else if (extent == (int64_t) c->out_in_chunk.size()) {
        for (size_t i = 0; i < c->out_in_chunk.size(); ++i) {
            rows_in_tensor.push_back((int64_t) i);
            positions.push_back(c->chunk_start + c->out_in_chunk[i]);
        }
    } else {
        c->skipped.push_back(full + " ne=[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
                             std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]");
        return true;
    }
    if (rows_in_tensor.empty()) return true;

    int64_t per = 1;
    for (int d = 0; d < std::min(tdim, 4); ++d) per *= t->ne[d];

    std::vector<char> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    // gather by strides (views need not be contiguous)
    std::vector<float> row(per);
    const std::string key = full + (occ ? "." + std::to_string(occ) : "");
    auto it = c->files.find(key);
    if (it == c->files.end()) {
        CaptureFile f;
        f.full = full; f.base = base; f.layer = layer; f.occurrence = occ;
        f.op = ggml_op_desc(t);
        for (int d = 0; d < 4; ++d) f.ne[d] = t->ne[d];
        f.cols = per;
        f.rel = "tensors/" + key + ".f32";
        f.fp = fopen((c->dir + "/" + f.rel).c_str(), "wb");
        if (!f.fp) { fprintf(stderr, "mfe-golden: cannot write %s\n", f.rel.c_str()); exit(1); }
        it = c->files.emplace(key, std::move(f)).first;
        c->order.push_back(key);
    }
    CaptureFile & f = it->second;
    if (f.cols != per) {
        c->skipped.push_back(key + " (cols changed " + std::to_string(f.cols) + " -> " + std::to_string(per) + ")");
        return true;
    }
    for (size_t k = 0; k < rows_in_tensor.size(); ++k) {
        const int64_t r = rows_in_tensor[k];
        int64_t idx = 0;
        // element (i0, i1, i2, i3) of row r: the token index sits at tdim
        int64_t lim[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        if (tdim < 4) lim[tdim] = 1;
        for (int64_t i3 = 0; i3 < lim[3]; ++i3)
        for (int64_t i2 = 0; i2 < lim[2]; ++i2)
        for (int64_t i1 = 0; i1 < lim[1]; ++i1)
        for (int64_t i0 = 0; i0 < lim[0]; ++i0) {
            int64_t ix[4] = {i0, i1, i2, i3};
            if (tdim < 4) ix[tdim] = r;
            const size_t off = ix[0] * t->nb[0] + ix[1] * t->nb[1] + ix[2] * t->nb[2] + ix[3] * t->nb[3];
            row[idx++] = *(const float *) (raw.data() + off);
        }
        if (fwrite(row.data(), sizeof(float), per, f.fp) != (size_t) per) {
            fprintf(stderr, "mfe-golden: short write to %s\n", f.rel.c_str()); exit(1);
        }
        f.positions.push_back(positions[k]);
    }
    return true;
}

// ---------------------------------------------------------------- perturbation

// --perturb: a controlled per-op error. In pass A, every F32 contiguous graph
// tensor whose base name is listed gets Gaussian noise of relative size
// `scale` added in place (x += scale * rms(row) * N(0, 1), so the tensor's
// relative L2 error is about `scale`), before the ops that read it run. Used
// to measure which per-op error level the D1 bar tolerates
// (docs/NATIVE_PARITY.md, "Per-op bounds").
struct Perturb {
    std::set<std::string> bases;   // "*" matches every name
    std::string op;                // optional: only nodes of this ggml op (e.g. MUL_MAT)
    double scale = 0;
    std::mt19937_64 rng{20261119};
    int64_t tensors = 0;
};

static bool perturb_cb(struct ggml_tensor * t, bool ask, void * user) {
    Perturb * pt = (Perturb *) user;
    std::string base; int layer;
    parse_name(t->name, base, layer);
    const bool named = pt->bases.count("*") || pt->bases.count(base);
    const bool want = named && (pt->op.empty() || pt->op == ggml_op_name(t->op)) &&
                      t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    if (ask) return want;
    if (!want) return true;
    const int64_t cols = t->ne[0], rows = ggml_nelements(t) / cols;
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    std::normal_distribution<double> n01(0.0, 1.0);
    for (int64_t r = 0; r < rows; ++r) {
        float * x = v.data() + r * cols;
        double ss = 0;
        for (int64_t i = 0; i < cols; ++i) ss += (double) x[i] * x[i];
        const double sigma = pt->scale * std::sqrt(ss / cols);
        for (int64_t i = 0; i < cols; ++i) x[i] = (float) (x[i] + sigma * n01(pt->rng));
    }
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    pt->tensors++;
    return true;
}

// ---------------------------------------------------------------- prompts

struct Prompt {
    std::string id;
    common_json messages;
    common_json tools;
    bool has_tools = false;
    bool enable_thinking = false;
    int max_new = 128;
    int sel_head = 16, sel_tail = 16, sel_stride = 0;
    int decode_capture = 8;
};

struct Options {
    std::string model, prompts, out, mode = "golden", backend = "metal";
    std::set<std::string> only;
    int n_batch = 2048, n_ubatch = 512, threads = 0, state_steps = 1;
    std::string kv = "f16", fa = "auto", teacher, perturb, perturb_op;
    double perturb_scale = 0;
    int max_new_override = -1;
    std::vector<NameSpec> specs;
};

static void add_specs(std::vector<NameSpec> & v, const std::string & list, bool all_rows, bool state_only) {
    for (const auto & item : split(list, ',')) {
        NameSpec s;
        const size_t h = item.find('#');
        s.base = item.substr(0, h);
        if (h != std::string::npos) s.occurrence = atoi(item.c_str() + h + 1);
        s.all_rows = all_rows;
        s.state_only = state_only;
        v.push_back(s);
    }
}

// The TapPoint tensors (docs/NATIVE_PARITY.md), the op boundaries the
// component tests feed and compare, and the state at the first decode step.
static const char * kDefaultSel =
    "model.input_embed,attn_norm,attn_output,linear_attn_out,ffn_out,l_out,result_norm,result_output,"
    "Qcur_full,Kcur,Vcur,attn_gated,linear_attn_qkv_mixed,z,alpha,beta,final_output,attn_post_norm,"
    "ffn_gate,ffn_up,ffn_swiglu";
static const char * kDefaultAll = "";
static const char * kDefaultState = "conv_states,state_predelta";

static int usage() {
    fprintf(stderr,
        "usage: mfe-golden --model <gguf> --prompts <set.json> --out <dir>\n"
        "         [--mode golden|logits] [--backend metal|cpu] [--only id,id]\n"
        "         [--batch 2048] [--ubatch 512] [--kv f16|f32] [--fa auto|on|off]\n"
        "         [--threads N] [--max-new N] [--sel names] [--all names] [--state names]\n"
        "         [--state-steps 1] [--teacher <golden run dir>]\n"
        "         [--perturb names|'*' --perturb-scale s [--perturb-op MUL_MAT]]  (pass A; with --mode logits --teacher)\n");
    return 2;
}

static int argmax(const float * p, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;  // ties: lowest id
    return best;
}

int main(int argc, char ** argv) {
    Options o;
    std::string sel = kDefaultSel, all = kDefaultAll, state = kDefaultState;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if (a == "--model") o.model = next();
        else if (a == "--prompts") o.prompts = next();
        else if (a == "--out") o.out = next();
        else if (a == "--mode") o.mode = next();
        else if (a == "--backend") o.backend = next();
        else if (a == "--only") { for (auto & s : split(next(), ',')) o.only.insert(s); }
        else if (a == "--batch") o.n_batch = atoi(next().c_str());
        else if (a == "--ubatch") o.n_ubatch = atoi(next().c_str());
        else if (a == "--kv") o.kv = next();
        else if (a == "--fa") o.fa = next();
        else if (a == "--threads") o.threads = atoi(next().c_str());
        else if (a == "--max-new") o.max_new_override = atoi(next().c_str());
        else if (a == "--sel") sel = next();
        else if (a == "--all") all = next();
        else if (a == "--state") state = next();
        else if (a == "--state-steps") o.state_steps = atoi(next().c_str());
        else if (a == "--teacher") o.teacher = next();
        else if (a == "--perturb") o.perturb = next();
        else if (a == "--perturb-scale") o.perturb_scale = atof(next().c_str());
        else if (a == "--perturb-op") o.perturb_op = next();
        else { fprintf(stderr, "mfe-golden: unknown argument %s\n", a.c_str()); return usage(); }
    }
    if (o.model.empty() || o.prompts.empty() || o.out.empty()) return usage();
    if (o.mode != "golden" && o.mode != "logits") return usage();
    if (o.backend != "metal" && o.backend != "cpu") return usage();
    add_specs(o.specs, all, true, false);
    add_specs(o.specs, sel, false, false);
    add_specs(o.specs, state, false, true);

    const common_json set = common_json::parse(read_file(o.prompts));
    std::vector<Prompt> prompts;
    const common_json & arr = set.at("prompts");
    for (size_t i = 0; i < arr.size(); ++i) {
        const common_json & p = arr.at(i);
        Prompt q;
        q.id = p.at("id").get<std::string>();
        if (!o.only.empty() && !o.only.count(q.id)) continue;
        q.messages = p.at("messages");
        if (p.contains("tools")) { q.tools = p.at("tools"); q.has_tools = true; }
        q.enable_thinking = p.value<bool>("enable_thinking", false);
        q.max_new = (int) p.value<int64_t>("max_new", 128);
        if (o.max_new_override >= 0) q.max_new = o.max_new_override;
        if (p.contains("capture")) {
            const common_json & c = p.at("capture");
            q.sel_head = (int) c.value<int64_t>("head", 16);
            q.sel_tail = (int) c.value<int64_t>("tail", 16);
            q.sel_stride = (int) c.value<int64_t>("stride", 0);
            q.decode_capture = (int) c.value<int64_t>("decode_steps", 8);
        }
        prompts.push_back(q);
    }
    if (prompts.empty()) { fprintf(stderr, "mfe-golden: no prompts selected\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.backend == "metal" ? -1 : 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.load_mtp = false;
    if (o.backend == "cpu") {
        // keep weights in the mmap (no anonymous CPU_REPACK copy), as the
        // product never repacks; the CPU run is a divergence reference only
        mp.use_extra_bufts = false;
        static ggml_backend_dev_t devs[2] = {nullptr, nullptr};
        devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        mp.devices = devs;
    }
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) { fprintf(stderr, "mfe-golden: load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto tmpls = common_chat_templates_init(model, "");

    char desc[256] = {0};
    llama_model_desc(model, desc, sizeof desc);
    mkdirs(o.out);

    for (const Prompt & q : prompts) {
        const double t_prompt0 = now_s();
        const std::string dir = o.out + "/" + q.id;
        mkdirs(dir + "/tensors");

        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(q.messages);
        if (q.has_tools) in.tools = common_chat_tools_parse_oaicompat(q.tools);
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.enable_thinking = q.enable_thinking;
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
        const std::string rendered = cp.prompt.str();
        write_file(dir + "/prompt.txt", rendered);

        int n = -llama_tokenize(vocab, rendered.data(), (int32_t) rendered.size(), nullptr, 0, true, true);
        std::vector<llama_token> toks(n);
        if (llama_tokenize(vocab, rendered.data(), (int32_t) rendered.size(), toks.data(), n, true, true) != n) {
            fprintf(stderr, "mfe-golden: tokenize mismatch for %s\n", q.id.c_str()); return 1;
        }
        const int n_prompt = n;

        // teacher forcing: the golden run's tokens drive every step
        std::vector<llama_token> teacher;
        if (!o.teacher.empty()) {
            const common_json t = common_json::parse(read_file(o.teacher + "/" + q.id + "/golden.json"));
            const common_json & tp = t.at("prompt_tokens");
            bool same = (int) tp.size() == n_prompt;
            for (int i = 0; same && i < n_prompt; ++i) same = (llama_token) tp.at(i).get<int64_t>() == toks[i];
            if (!same) { fprintf(stderr, "mfe-golden: %s: prompt tokens differ from the teacher's\n", q.id.c_str()); return 1; }
            const common_json & tg = t.at("generated_tokens");
            for (size_t i = 0; i < tg.size(); ++i) teacher.push_back((llama_token) tg.at(i).get<int64_t>());
            if (teacher.empty()) { fprintf(stderr, "mfe-golden: %s: teacher has no tokens\n", q.id.c_str()); return 1; }
        }
        const int max_new = teacher.empty() ? q.max_new : (int) teacher.size();
        const int n_ctx = ((n_prompt + max_new + 1 + 255) / 256) * 256;

        Perturb perturb;
        for (const auto & b : split(o.perturb, ',')) perturb.bases.insert(b);
        perturb.scale = o.perturb_scale;
        perturb.op = o.perturb_op;
        auto make_ctx = [&](int n_batch, int n_ubatch, Capture * cap) {
            llama_context_params p = llama_context_default_params();
            p.n_ctx = n_ctx; p.n_batch = n_batch; p.n_ubatch = n_ubatch; p.n_seq_max = 1;
            p.no_perf = true;
            if (o.threads > 0) { p.n_threads = o.threads; p.n_threads_batch = o.threads; }
            p.type_k = p.type_v = o.kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
            p.flash_attn_type = o.fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                              : o.fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
            if (cap) { p.cb_eval = eval_cb; p.cb_eval_user_data = cap; }
            else if (!perturb.bases.empty()) { p.cb_eval = perturb_cb; p.cb_eval_user_data = &perturb; }
            llama_context * c = llama_init_from_model(model, p);
            if (!c) { fprintf(stderr, "mfe-golden: context failed\n"); exit(1); }
            return c;
        };

        // ---------------- pass A: the golden logits and greedy tokens
        llama_context * ctx = make_ctx(o.n_batch, o.n_ubatch, nullptr);
        std::vector<float> logits;  // [n_gen x n_vocab]
        std::vector<llama_token> gen, own;
        bool stopped_eog = false;
        const double ta0 = now_s();
        double t_prefill = 0;
        {
            llama_batch b = llama_batch_init(o.n_batch, 0, 1);
            for (int s = 0; s < n_prompt; s += o.n_batch) {
                const int m = std::min(o.n_batch, n_prompt - s);
                b.n_tokens = m;
                for (int i = 0; i < m; ++i) {
                    b.token[i] = toks[s + i]; b.pos[i] = s + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
                    b.logits[i] = (s + i == n_prompt - 1);
                }
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "mfe-golden: prefill failed\n"); return 1; }
            }
            llama_synchronize(ctx);  // llama_decode may return before the GPU finishes
            t_prefill = now_s() - ta0;
            int pos = n_prompt;
            for (int g = 0; g < max_new; ++g) {
                const float * row = llama_get_logits_ith(ctx, -1);
                logits.insert(logits.end(), row, row + n_vocab);
                const llama_token mine = argmax(row, n_vocab);
                const llama_token t = teacher.empty() ? mine : teacher[g];
                own.push_back(mine);
                gen.push_back(t);
                if (llama_vocab_is_eog(vocab, t)) { stopped_eog = true; break; }
                if (g + 1 == max_new) break;
                b.n_tokens = 1;
                b.token[0] = t; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "mfe-golden: decode failed\n"); return 1; }
            }
            llama_batch_free(b);
        }
        const double t_pass_a = now_s() - ta0;
        llama_free(ctx);
        {
            FILE * f = fopen((dir + "/logits.f32").c_str(), "wb");
            if (!f) { fprintf(stderr, "mfe-golden: cannot write %s/logits.f32\n", dir.c_str()); return 1; }
            const bool ok = fwrite(logits.data(), sizeof(float), logits.size(), f) == logits.size();
            if (fclose(f) != 0 || !ok) { fprintf(stderr, "mfe-golden: short write to %s/logits.f32\n", dir.c_str()); return 1; }
        }
        std::vector<int64_t> logit_pos;
        for (size_t g = 0; g < gen.size(); ++g) logit_pos.push_back(n_prompt - 1 + (int64_t) g);

        // ---------------- pass B: per-op tensors, teacher-forced
        Capture cap;
        cap.dir = dir;
        cap.specs = o.specs;
        std::vector<int64_t> sel_pos;
        double t_pass_b = 0;
        int decode_steps = 0;
        if (o.mode == "golden") {
            std::set<int64_t> s;
            for (int i = 0; i < std::min(q.sel_head, n_prompt); ++i) s.insert(i);
            for (int i = std::max(0, n_prompt - q.sel_tail); i < n_prompt; ++i) s.insert(i);
            if (q.sel_stride > 0) for (int i = 0; i < n_prompt; i += q.sel_stride) s.insert(i);
            sel_pos.assign(s.begin(), s.end());
            decode_steps = std::min<int>(q.decode_capture, (int) gen.size() - 1);
            if (decode_steps < 0) decode_steps = 0;

            const double tb0 = now_s();
            ctx = make_ctx(o.n_ubatch, o.n_ubatch, &cap);
            llama_batch b = llama_batch_init(o.n_ubatch, 0, 1);
            for (int st = 0; st < n_prompt; st += o.n_ubatch) {
                const int m = std::min(o.n_ubatch, n_prompt - st);
                cap.chunk_start = st; cap.chunk_rows = m; cap.seen.clear(); cap.state_step = false;
                cap.sel_in_chunk.clear(); cap.out_in_chunk.clear();
                b.n_tokens = m;
                for (int i = 0; i < m; ++i) {
                    const bool keep = s.count(st + i) > 0;
                    if (keep) { cap.sel_in_chunk.push_back(i); cap.out_in_chunk.push_back(i); }
                    b.token[i] = toks[st + i]; b.pos[i] = st + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
                    b.logits[i] = keep;
                }
                if (cap.out_in_chunk.empty()) {  // llama.cpp needs one output; keep the last row
                    b.logits[m - 1] = 1; cap.out_in_chunk.push_back(m - 1);
                }
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "mfe-golden: capture prefill failed\n"); return 1; }
            }
            for (int g = 0; g < decode_steps; ++g) {
                cap.chunk_start = n_prompt + g; cap.chunk_rows = 1; cap.seen.clear();
                cap.state_step = g < o.state_steps;
                cap.sel_in_chunk = {0}; cap.out_in_chunk = {0};
                b.n_tokens = 1;
                b.token[0] = gen[g]; b.pos[0] = n_prompt + g; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "mfe-golden: capture decode failed\n"); return 1; }
            }
            llama_batch_free(b);
            llama_free(ctx);
            t_pass_b = now_s() - tb0;
            for (auto & kv : cap.files) {
                if (fclose(kv.second.fp) != 0) { fprintf(stderr, "mfe-golden: cannot close %s\n", kv.second.rel.c_str()); return 1; }
                kv.second.fp = nullptr;
            }
        }

        // ---------------- manifest
        std::string m = "{\n";
        m += "  \"format\": \"mfe-golden-v1\",\n";
        m += "  \"prompt_id\": " + json_str(q.id) + ",\n";
        m += "  \"prompt_set\": " + json_str(set.value("name", "")) + ",\n";
        m += "  \"mode\": " + json_str(o.mode) + ",\n";
        m += "  \"source\": {\"engine\": \"llama.cpp\", \"commit\": \"b1b770d90d94cad9e9ea5b993bdbc383a522fb28\", "
             "\"backend\": " + json_str(o.backend) + ", \"model_desc\": " + json_str(desc) +
             ", \"model_path\": " + json_str(o.model) + ", \"n_batch\": " + std::to_string(o.n_batch) +
             ", \"n_ubatch\": " + std::to_string(o.n_ubatch) + ", \"kv\": " + json_str(o.kv) +
             ", \"flash_attn\": " + json_str(o.fa) + ", \"threads\": " + std::to_string(o.threads) +
             ", \"n_ctx\": " + std::to_string(n_ctx) + "},\n";
        m += "  \"enable_thinking\": " + std::string(q.enable_thinking ? "true" : "false") + ",\n";
        m += "  \"rendered_prompt_file\": \"prompt.txt\",\n";
        m += "  \"n_vocab\": " + std::to_string(n_vocab) + ",\n";
        m += "  \"prompt_tokens\": " + json_ints(toks) + ",\n";
        m += "  \"generated_tokens\": " + json_ints(gen) + ",\n";
        m += "  \"argmax_tokens\": " + json_ints(own) + ",\n";
        m += "  \"teacher\": " + json_str(o.teacher) + ",\n";
        m += "  \"perturb\": {\"names\": " + json_str(o.perturb) + ", \"op\": " + json_str(o.perturb_op) +
             ", \"scale\": " + std::to_string(o.perturb_scale) +
             ", \"tensors\": " + std::to_string(perturb.tensors) + "},\n";
        m += "  \"stopped_on_eog\": " + std::string(stopped_eog ? "true" : "false") + ",\n";
        m += "  \"logits\": {\"file\": \"logits.f32\", \"dtype\": \"f32\", \"rows\": " + std::to_string(gen.size()) +
             ", \"cols\": " + std::to_string(n_vocab) + ", \"position_ranges\": " + json_ranges(logit_pos) + "},\n";
        m += "  \"timing_s\": {\"prefill\": " + std::to_string(t_prefill) + ", \"pass_a\": " + std::to_string(t_pass_a) +
             ", \"pass_b\": " + std::to_string(t_pass_b) + "},\n";
        m += "  \"capture\": {\"chunk\": " + std::to_string(o.n_ubatch) + ", \"selected_ranges\": " + json_ranges(sel_pos) +
             ", \"decode_steps\": " + std::to_string(decode_steps) + ", \"state_steps\": " + std::to_string(o.state_steps) + "},\n";
        m += "  \"tensors\": [";
        bool first = true;
        for (const auto & key : cap.order) {
            const CaptureFile & f = cap.files.at(key);
            m += first ? "\n" : ",\n";
            first = false;
            m += "    {\"key\": " + json_str(key) + ", \"name\": " + json_str(f.full) + ", \"base\": " + json_str(f.base) +
                 ", \"layer\": " + std::to_string(f.layer) + ", \"occurrence\": " + std::to_string(f.occurrence) +
                 ", \"op\": " + json_str(f.op) + ", \"ne\": [" + std::to_string(f.ne[0]) + "," + std::to_string(f.ne[1]) +
                 "," + std::to_string(f.ne[2]) + "," + std::to_string(f.ne[3]) + "], \"dtype\": \"f32\", \"cols\": " +
                 std::to_string(f.cols) + ", \"rows\": " + std::to_string(f.positions.size()) +
                 ", \"position_ranges\": " + json_ranges(f.positions) + ", \"file\": " + json_str(f.rel) + "}";
        }
        m += "\n  ],\n";
        m += "  \"skipped\": [";
        for (size_t i = 0; i < cap.skipped.size() && i < 64; ++i) { if (i) m += ","; m += json_str(cap.skipped[i]); }
        m += "]\n}\n";
        write_file(dir + "/golden.json", m);
        fprintf(stderr, "mfe-golden: %s prompt=%d gen=%zu eog=%d tensors=%zu skipped=%zu prefill=%.2fs passA=%.2fs passB=%.2fs total=%.2fs\n",
                q.id.c_str(), n_prompt, gen.size(), (int) stopped_eog, cap.order.size(), cap.skipped.size(),
                t_prefill, t_pass_a, t_pass_b, now_s() - t_prompt0);
    }
    tmpls.reset();
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
