// mfe-bench: end-to-end llama.cpp baseline at the pin (b1b770d9) for the
// native engine's benchmark protocol (docs/NATIVE_BENCHMARKS.md).
//
//   mfe-bench --model <gguf> --tokens <golden.json> [--ctx 512,4096,8192]
//             [--gen 128] [--suffix 64] [--reps 3] [--json <file>]
//
// Settings are the product LlamaBackend's: every layer on Metal, n_batch 2048,
// n_ubatch 512, F16 KV, flash attention auto, one sequence. Prompt tokens are
// real text: the given golden's prompt tokens (long-32k), repeated when a
// context needs more. For each context N and repetition, on a fresh context:
//   cold TTFT  N prompt tokens prefilled, then the first token's logits read
//              and argmax taken; prefill tok/s = N / cold TTFT
//   warm TTFT  a follow-up of --suffix tokens appended to the cached N-token
//              history, then the first token (prefix reuse: only the suffix
//              is prefilled)
//   decode     --gen greedy tokens after that, one per step: tok/s at depth N
// Load time (model and context) is reported once. One untimed warm-up per
// context, then medians over --reps.
#include "llama.h"
#include "json.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

static int argmax(const float * p, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[b]) b = i;
    return b;
}

static bool decode_all(llama_context * ctx, llama_batch & b, const std::vector<llama_token> & toks, int start, int n_batch) {
    for (int s = 0; s < (int) toks.size(); s += n_batch) {
        const int m = std::min(n_batch, (int) toks.size() - s);
        b.n_tokens = m;
        for (int i = 0; i < m; ++i) {
            b.token[i] = toks[s + i]; b.pos[i] = start + s + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (s + i == (int) toks.size() - 1);
        }
        if (llama_decode(ctx, b) != 0) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, tokens_path, json_path;
    std::vector<int> ctxs = {512, 4096, 8192};
    int gen = 128, suffix = 64, reps = 3;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--model") model_path = next();
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--json") json_path = next();
        else if (a == "--gen") gen = atoi(next().c_str());
        else if (a == "--suffix") suffix = atoi(next().c_str());
        else if (a == "--reps") reps = atoi(next().c_str());
        else if (a == "--ctx") { ctxs.clear(); std::stringstream ss(next()); std::string x; while (std::getline(ss, x, ',')) ctxs.push_back(atoi(x.c_str())); }
        else { fprintf(stderr, "mfe-bench: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || tokens_path.empty()) { fprintf(stderr, "usage: mfe-bench --model <gguf> --tokens <golden.json> ...\n"); return 2; }
    std::ifstream tf(tokens_path); std::stringstream tss; tss << tf.rdbuf();
    const common_json gj = common_json::parse(tss.str());
    std::vector<llama_token> source;
    const common_json & pt = gj.at("prompt_tokens");
    for (size_t i = 0; i < pt.size(); ++i) source.push_back((llama_token) pt.at(i).get<int64_t>());
    if (source.empty()) { fprintf(stderr, "mfe-bench: no tokens\n"); return 2; }
    auto take = [&](int start, int n) {
        std::vector<llama_token> v(n);
        for (int i = 0; i < n; ++i) v[i] = source[(start + i) % source.size()];
        return v;
    };

    const double tl0 = now_s();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1; mp.load_mode = LLAMA_LOAD_MODE_MMAP; mp.load_mtp = false;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "mfe-bench: load failed\n"); return 1; }
    const double load_s = now_s() - tl0;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_batch = 2048, n_ubatch = 512;

    std::string js = "{\n \"tool\": \"mfe-bench\", \"llama_commit\": \"b1b770d90d94cad9e9ea5b993bdbc383a522fb28\", \"backend\": \"metal\", "
                     "\"n_batch\": 2048, \"n_ubatch\": 512, \"kv\": \"f16\", \"flash_attn\": \"auto\", \"model_load_s\": " + std::to_string(load_s) +
                     ", \"gen\": " + std::to_string(gen) + ", \"suffix\": " + std::to_string(suffix) + ", \"reps\": " + std::to_string(reps) + ",\n \"results\": [\n";
    printf("| context | cold TTFT s | prefill tok/s | warm TTFT s (+%d) | decode tok/s at depth | context create s |\n|---:|---:|---:|---:|---:|---:|\n", suffix);
    for (size_t ci = 0; ci < ctxs.size(); ++ci) {
        const int N = ctxs[ci];
        std::vector<double> cold, warm, dec, create;
        // r = -1 is an untimed warm-up (first-use shader compiles, page-ins)
        for (int r = -1; r < reps; ++r) {
            const double tc0 = now_s();
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = ((N + suffix + gen + 1 + 255) / 256) * 256;
            cp.n_batch = n_batch; cp.n_ubatch = n_ubatch; cp.n_seq_max = 1; cp.no_perf = true;
            llama_context * ctx = llama_init_from_model(model, cp);
            if (!ctx) { fprintf(stderr, "mfe-bench: context failed\n"); return 1; }
            create.push_back(now_s() - tc0);
            llama_batch b = llama_batch_init(n_batch, 0, 1);
            const std::vector<llama_token> prompt = take(0, N), follow = take(N, suffix);

            double t0 = now_s();
            if (!decode_all(ctx, b, prompt, 0, n_batch)) { fprintf(stderr, "mfe-bench: prefill failed\n"); return 1; }
            int tok = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            cold.push_back(now_s() - t0);

            t0 = now_s();
            if (!decode_all(ctx, b, follow, N, n_batch)) { fprintf(stderr, "mfe-bench: suffix failed\n"); return 1; }
            tok = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            warm.push_back(now_s() - t0);

            int pos = N + suffix;
            t0 = now_s();
            for (int g = 0; g < gen; ++g) {
                b.n_tokens = 1; b.token[0] = tok; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "mfe-bench: decode failed\n"); return 1; }
                tok = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            }
            dec.push_back(gen / (now_s() - t0));
            llama_batch_free(b);
            llama_free(ctx);
            if (r < 0) {  // the warm-up is not a measurement
                cold.pop_back(); warm.pop_back(); dec.pop_back(); create.pop_back();
                fprintf(stderr, "mfe-bench: N=%d warm-up done\n", N);
                continue;
            }
            fprintf(stderr, "mfe-bench: N=%d rep=%d cold=%.3fs warm=%.3fs decode=%.1f tok/s\n", N, r, cold.back(), warm.back(), dec.back());
        }
        const double c = median(cold), w = median(warm), d = median(dec), k = median(create);
        printf("| %d | %.3f | %.1f | %.3f | %.1f | %.3f |\n", N, c, N / c, w, d, k);
        char line[512];
        snprintf(line, sizeof line, "  {\"context\": %d, \"cold_ttft_s\": %.4f, \"prefill_tok_s\": %.2f, \"warm_ttft_s\": %.4f, "
                 "\"decode_tok_s\": %.2f, \"context_create_s\": %.4f, \"cold_min\": %.4f, \"cold_max\": %.4f, \"decode_min\": %.2f, \"decode_max\": %.2f}%s\n",
                 N, c, N / c, w, d, k, *std::min_element(cold.begin(), cold.end()), *std::max_element(cold.begin(), cold.end()),
                 *std::min_element(dec.begin(), dec.end()), *std::max_element(dec.begin(), dec.end()), ci + 1 < ctxs.size() ? "," : "");
        js += line;
    }
    js += " ]\n}\n";
    if (!json_path.empty()) { std::ofstream f(json_path); f << js; }
    printf("\nmodel load %.3f s\n", load_s);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
