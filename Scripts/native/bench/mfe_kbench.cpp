// mfe-kbench: ggml-metal kernel baselines at the pin (b1b770d9) for the
// native engine's benchmark gates (docs/NATIVE_BENCHMARKS.md).
//
//   mfe-kbench [--suite dispatch,copy,gemm,attn,gdn] [--types q4_K,...] [--rows 1,4,8,16,512]
//              [--shapes K:N,...] [--ctx 512,4096,8192] [--reps 20] [--min-mib 256] [--max-mib 2048]
//              [--json <file>]
//
// Every measurement builds one ggml graph of M independent copies of the op
// on the Metal backend, with concurrent dispatch off
// (GGML_METAL_CONCURRENCY_DISABLE), as a model graph's dependent ops run. The
// copies use distinct weight, KV or state buffers. M is sized from each copy's
// own working set so the graph's distinct bytes reach at least --min-mib, past
// the system cache, and its total allocation stays within --max-mib (bounded
// memory). A GEMM's activation is shared by every copy.
//
// Timing is host wall-clock around the synchronous ggml_backend_graph_compute
// (ggml-metal exposes no GPU timestamps): time per op = median graph time / M
// over --reps runs, after 3 warm-ups. Each op therefore includes its share of
// dispatch cost; the "dispatch" suite reports that floor. W2 and W3 must time
// their kernels the same way (M copies in one synchronous submission, host
// clock) for the >= ggml-metal gates to compare like with like.
//
// Reported per op: microseconds, bytes the op must read (weights or KV/state
// plus activations), the bandwidth that implies, and the share of the
// memory-bandwidth floor (bytes / peak) against the nominal peak and against
// this run's measured copy bandwidth. No model is loaded.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> v; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, sep)) if (!x.empty()) v.push_back(x);
    return v;
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Options {
    std::vector<std::string> suites = {"dispatch", "copy", "gemm", "attn", "gdn"};
    std::vector<std::string> types;   // empty: Quick's
    std::vector<int> rows = {1, 2, 4, 8, 16, 512};
    std::vector<std::pair<int, int>> shapes;  // empty: per type, Quick's
    std::vector<int> ctx = {512, 4096, 8192};
    int reps = 20;
    double min_mib = 256;   // per-graph distinct working set, past the system cache
    double max_mib = 2048;  // cap on a graph's total allocation (bounded memory)
    double nominal_gbs = 400;
    std::string json;
};

struct Result {
    std::string suite, name, type;
    int k = 0, n = 0, rows = 0, ctx = 0, copies = 0;
    double us = 0, bytes = 0;
};

static std::vector<Result> g_results;

// Copies of an op whose distinct per-copy working set is `per_copy` bytes:
// enough to pass --min-mib (past the system cache), never more than
// --max-mib in all (bounded memory). `shared` bytes are allocated once.
static int copies_for(const Options & o, double per_copy, double shared = 0) {
    const double mib = 1 << 20;
    int c = std::max(1, (int) std::ceil(o.min_mib * mib / per_copy));
    const int cap = std::max(1, (int) std::floor((o.max_mib * mib - shared) / per_copy));
    return std::min(c, cap);
}
static double g_copy_gbs = 0;

// Runs `build` (which adds one op copy per call into ctx and returns its
// output) M times into a graph, allocates, initializes with `init`, and times.
template <typename Build, typename Init>
static double time_graph(ggml_backend_t be, int copies, int reps, size_t ctx_tensors, Build build, Init init) {
    // graph capacity scales with the copies (each copy adds a few nodes and leafs)
    const size_t graph_size = std::max<size_t>(4096, ctx_tensors * (size_t) copies * 2 + 64);
    ggml_init_params ip = { ggml_tensor_overhead() * (ctx_tensors * copies + 16) + ggml_graph_overhead_custom(graph_size, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
    std::vector<ggml_tensor *> outs;
    for (int i = 0; i < copies; ++i) {
        ggml_tensor * out = build(ctx, i);
        ggml_build_forward_expand(gf, out);
        outs.push_back(out);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "mfe-kbench: allocation failed\n"); exit(1); }
    init(ctx);
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, gf);
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_us();
        ggml_backend_graph_compute(be, gf);
        t.push_back(now_us() - t0);
    }
    std::sort(t.begin(), t.end());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return t[t.size() / 2] / copies;
}

static std::mt19937 g_rng(20261119);

static void fill_f32(std::vector<float> & v, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    for (auto & x : v) x = d(g_rng);
}

static void set_random(ggml_tensor * t, float scale = 1.0f) {
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> v(ggml_nelements(t)); fill_f32(v, scale);
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<float> v(ggml_nelements(t)); fill_f32(v, scale);
        std::vector<ggml_fp16_t> h(v.size());
        ggml_fp32_to_fp16_row(v.data(), h.data(), (int64_t) v.size());
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * 2);
    } else {
        std::vector<float> zero(ggml_nelements(t), 0.0f);
        ggml_backend_tensor_set(t, zero.data(), 0, zero.size() * 4);
    }
}

// Quantized weight bytes for type `t`, [K x N], from random F32 rows. One
// matrix is quantized and copied into every copy (contents do not change
// the bandwidth a kernel needs).
static std::vector<uint8_t> quantized(ggml_type t, int K, int N) {
    std::vector<float> f((size_t) K * N); fill_f32(f, 0.02f);
    std::vector<uint8_t> q(ggml_row_size(t, K) * N);
    if (t == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) q.data(), (int64_t) f.size());
    } else if (t == GGML_TYPE_F32) {
        memcpy(q.data(), f.data(), q.size());
    } else {
        std::vector<float> imatrix;
        const float * im = nullptr;
        if (ggml_quantize_requires_imatrix(t)) { imatrix.assign(K, 1.0f); im = imatrix.data(); }
        ggml_quantize_init(t);
        ggml_quantize_chunk(t, f.data(), q.data(), 0, N, K, im);
    }
    return q;
}

// ---------------------------------------------------------------- suites

// The host-timing floor: 1024 copies of a one-element op. Every figure here is
// host wall-clock around a synchronous graph compute (ggml-metal exposes no
// GPU timestamps), so each op carries about this much dispatch cost; W2/W3
// time their kernels the same way (docs/NATIVE_BENCHMARKS.md).
static void bench_dispatch(ggml_backend_t be, const Options & o) {
    std::vector<ggml_tensor *> xs;
    const int copies = 1024;
    const double us = time_graph(be, copies, o.reps, 2,
        [&](ggml_context * ctx, int) {
            ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            xs.push_back(x);
            return ggml_scale(ctx, x, 2.0f);
        },
        [&](ggml_context *) { for (auto * x : xs) set_random(x); });
    Result r; r.suite = "dispatch"; r.name = "scale 1 element (dispatch floor)"; r.type = "f32";
    r.copies = copies; r.us = us; r.bytes = 8;
    g_results.push_back(r);
    fprintf(stderr, "dispatch floor: %.2f us per op\n", us);
}

static void bench_copy(ggml_backend_t be, const Options & o) {
    const int64_t n = (int64_t) 256 << 20;  // 256 Mi floats, 1 GiB
    const double us = time_graph(be, 1, o.reps, 2,
        [&](ggml_context * ctx, int) {
            ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            ggml_set_name(a, "src");
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            return ggml_cpy(ctx, a, b);
        },
        [&](ggml_context *) {});
    const double bytes = 2.0 * n * 4;
    g_copy_gbs = bytes / us / 1e3;
    Result r; r.suite = "copy"; r.name = "f32 copy 1 GiB"; r.type = "f32"; r.us = us; r.bytes = bytes; r.copies = 1;
    g_results.push_back(r);
    fprintf(stderr, "copy: %.1f us for %.0f MiB moved -> %.1f GB/s\n", us, bytes / (1 << 20), g_copy_gbs);
}

static std::map<std::string, std::vector<std::pair<int, int>>> quick_shapes() {
    // (K, N) per type where Quick has it (docs/NATIVE_ENGINE.md, tensor table)
    return {
        {"q4_K", {{2560, 8192}, {2560, 1024}, {4096, 2560}, {2560, 9216}, {9216, 2560}}},
        {"q5_K", {{2560, 8192}, {2560, 1024}, {2560, 4096}, {2560, 9216}, {9216, 2560}}},
        {"q6_K", {{2560, 1024}, {9216, 2560}, {2560, 248320}}},
        {"q8_0", {{4096, 2560}}},
        {"iq4_xs", {{2560, 9216}}},
        {"f16", {{2560, 32}}},
    };
}

static ggml_type type_of(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * name = ggml_type_name((ggml_type) t);
        if (name && s == name) return (ggml_type) t;
    }
    fprintf(stderr, "mfe-kbench: unknown type %s\n", s.c_str()); exit(2);
}

static void bench_gemm(ggml_backend_t be, const Options & o) {
    auto table = quick_shapes();
    std::vector<std::string> types = o.types;
    if (types.empty()) types = {"q4_K", "q5_K", "q6_K", "q8_0", "iq4_xs", "f16"};
    for (const auto & ts : types) {
        const ggml_type t = type_of(ts);
        auto shapes = o.shapes.empty() ? table[ts] : o.shapes;
        for (auto [K, N] : shapes) {
            const std::vector<uint8_t> w = quantized(t, K, N);
            const double wbytes = (double) w.size();
            for (int rows : o.rows) {
                // one activation x shared by every copy (as every layer reads
                // its own small input); distinct weights and outputs per copy
                const double outbytes = 4.0 * N * rows, xbytes = 4.0 * K * rows;
                const int copies = copies_for(o, wbytes + outbytes, xbytes);
                std::vector<ggml_tensor *> ws, xs;
                ggml_tensor * xshared = nullptr;
                const double us = time_graph(be, copies, o.reps, 3,
                    [&](ggml_context * ctx, int) {
                        ggml_tensor * W = ggml_new_tensor_2d(ctx, t, K, N);
                        if (!xshared) { xshared = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, rows); xs.push_back(xshared); }
                        ggml_tensor * x = xshared;
                        ws.push_back(W);
                        return ggml_mul_mat(ctx, W, x);
                    },
                    [&](ggml_context *) {
                        for (auto * W : ws) ggml_backend_tensor_set(W, w.data(), 0, w.size());
                        for (auto * x : xs) set_random(x);
                    });
                Result r; r.suite = "gemm"; r.type = ts; r.k = K; r.n = N; r.rows = rows; r.copies = copies;
                r.us = us; r.bytes = wbytes + xbytes + outbytes;
                r.name = ts + " " + std::to_string(K) + "x" + std::to_string(N);
                g_results.push_back(r);
                fprintf(stderr, "gemm %-7s K=%5d N=%6d rows=%3d: %9.1f us  %6.1f GB/s\n", ts.c_str(), K, N, rows, us, r.bytes / us / 1e3);
            }
        }
    }
}

// Quick's full attention: 16 q heads, 4 KV heads, head 256, F16 KV, flash
// attention, causal F16 mask, as llama.cpp builds it (build_attn_mha).
static void bench_attn(ggml_backend_t be, const Options & o) {
    const int D = 256, HQ = 16, HKV = 4;
    for (int ctxlen : o.ctx) {
        for (int rows : o.rows) {
            if (rows > ctxlen) continue;
            const double kvbytes = 2.0 * ctxlen * HKV * D * 2;
            // per copy: K and V, q, mask and output (all distinct)
            const int copies = copies_for(o, kvbytes + 4.0 * rows * HQ * D * 2 + 2.0 * rows * ctxlen);
            std::vector<ggml_tensor *> qs, ks, vs, ms;
            const double us = time_graph(be, copies, o.reps, 10,
                [&](ggml_context * ctx, int) {
                    // llama.cpp's layouts: q [D, heads, tokens] and the KV cache
                    // [D, kv_heads, cells], each permuted to [D, tokens, heads]
                    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HQ, rows);
                    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, HKV, ctxlen);
                    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, HKV, ctxlen);
                    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ctxlen, rows);
                    qs.push_back(q); ks.push_back(k); vs.push_back(v); ms.push_back(m);
                    ggml_tensor * out = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3),
                        ggml_permute(ctx, k, 0, 2, 1, 3), ggml_permute(ctx, v, 0, 2, 1, 3), m,
                        1.0f / sqrtf((float) D), 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_n_kv_max(out, ctxlen);
                    ggml_prec_set_acc(out, GGML_PREC_F32);
                    return out;
                },
                [&](ggml_context *) {
                    for (auto * t : qs) set_random(t);
                    for (auto * t : ks) set_random(t, 0.5f);
                    for (auto * t : vs) set_random(t, 0.5f);
                    // causal: row r (at position ctxlen - rows + r) sees keys 0..ctxlen-rows+r
                    std::vector<ggml_fp16_t> mask((size_t) ctxlen * rows);
                    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
                    for (int r = 0; r < rows; ++r)
                        for (int c = 0; c < ctxlen; ++c) mask[(size_t) r * ctxlen + c] = c <= ctxlen - rows + r ? zero : ninf;
                    for (auto * t : ms) ggml_backend_tensor_set(t, mask.data(), 0, mask.size() * 2);
                });
            Result r; r.suite = "attn"; r.type = "f16kv"; r.ctx = ctxlen; r.rows = rows; r.copies = copies; r.us = us;
            r.bytes = kvbytes + 4.0 * rows * HQ * D * 2 + 2.0 * rows * ctxlen;
            r.name = "flash_attn_ext D256 HQ16 HKV4";
            g_results.push_back(r);
            fprintf(stderr, "attn ctx=%6d rows=%3d: %9.1f us  %6.1f GB/s\n", ctxlen, rows, us, r.bytes / us / 1e3);
        }
    }
}

// Quick's Gated DeltaNet core: conv1d (kernel 4, 8192 channels) + SiLU is
// separate; this times ggml_gated_delta_net (16 q/k heads, 32 v heads, d 128,
// scalar gate, F32 state) and ggml_ssm_conv for `rows` tokens.
static void bench_gdn(ggml_backend_t be, const Options & o) {
    const int S = 128, HK = 16, HV = 32, C = 8192, KC = 4;
    for (int rows : o.rows) {
        const double sbytes = 2.0 * S * S * HV * 4;  // read and write the state
        // per copy: state in and out, q/k/v/g/b and the output
        const int copies = copies_for(o, sbytes + 4.0 * rows * (2 * S * HK + 2 * S * HV + 2 * HV) + 4.0 * S * HV * rows);
        std::vector<ggml_tensor *> all;
        const double us = time_graph(be, copies, o.reps, 8,
            [&](ggml_context * ctx, int) {
                ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, rows, 1);
                ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, rows, 1);
                ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HV, rows, 1);
                ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, rows, 1);
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, rows, 1);
                ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, 1);
                all.insert(all.end(), {q, k, v, g, b, s});
                return ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
            },
            [&](ggml_context *) { for (auto * t : all) set_random(t, 0.1f); });
        Result r; r.suite = "gdn"; r.type = "f32"; r.rows = rows; r.copies = copies; r.us = us;
        r.bytes = sbytes + 4.0 * rows * (2 * S * HK + 2 * S * HV + 2 * HV);
        r.name = "gated_delta_net S128 HK16 HV32";
        g_results.push_back(r);
        fprintf(stderr, "gdn  rows=%3d: %9.1f us  %6.1f GB/s\n", rows, us, r.bytes / us / 1e3);

        // the convolution's own working set sets its copy count (input, weights
        // and output of each copy are distinct buffers), so it too reaches
        // --min-mib and cannot stay cache-resident
        const double cbytes = 4.0 * C * (KC - 1 + rows) + 4.0 * C * KC + 4.0 * C * rows;
        const int ccopies = copies_for(o, cbytes);
        std::vector<ggml_tensor *> cv;
        const double cus = time_graph(be, ccopies, o.reps, 3,
            [&](ggml_context * ctx, int) {
                ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KC - 1 + rows, C, 1);
                ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, KC, C);
                cv.insert(cv.end(), {x, w});
                return ggml_ssm_conv(ctx, x, w);
            },
            [&](ggml_context *) { for (auto * t : cv) set_random(t, 0.1f); });
        Result c; c.suite = "gdn"; c.type = "f32"; c.rows = rows; c.copies = ccopies; c.us = cus;
        c.bytes = cbytes;
        c.name = "ssm_conv K4 C8192";
        g_results.push_back(c);
        fprintf(stderr, "conv rows=%3d: %9.1f us\n", rows, cus);
    }
}

int main(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        auto ints = [&](const std::string & s) { std::vector<int> v; for (auto & x : split(s, ',')) v.push_back(atoi(x.c_str())); return v; };
        if (a == "--suite") o.suites = split(next(), ',');
        else if (a == "--types") o.types = split(next(), ',');
        else if (a == "--rows") o.rows = ints(next());
        else if (a == "--ctx") o.ctx = ints(next());
        else if (a == "--reps") o.reps = atoi(next().c_str());
        else if (a == "--min-mib") o.min_mib = atof(next().c_str());
        else if (a == "--max-mib") o.max_mib = atof(next().c_str());
        else if (a == "--nominal-gbs") o.nominal_gbs = atof(next().c_str());
        else if (a == "--json") o.json = next();
        else if (a == "--shapes") {
            for (auto & s : split(next(), ',')) {
                auto kn = split(s, ':');
                o.shapes.push_back({atoi(kn[0].c_str()), atoi(kn[1].c_str())});
            }
        }
        else { fprintf(stderr, "mfe-kbench: unknown argument %s\n", a.c_str()); return 2; }
    }
    setenv("GGML_METAL_CONCURRENCY_DISABLE", "1", 1);
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_metal_init();
    if (!be) { fprintf(stderr, "mfe-kbench: no Metal backend\n"); return 1; }
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    const char * desc = ggml_backend_dev_description(dev);
    int family = 0;
    for (int f = 10; f >= 1; --f) if (ggml_backend_metal_supports_family(be, f)) { family = f; break; }
    fprintf(stderr, "device: %s, MTLGPUFamilyApple%d\n", desc ? desc : "?", family);

    for (const auto & s : o.suites) {
        if (s == "dispatch") bench_dispatch(be, o);
        else if (s == "copy") bench_copy(be, o);
        else if (s == "gemm") bench_gemm(be, o);
        else if (s == "attn") bench_attn(be, o);
        else if (s == "gdn") bench_gdn(be, o);
        else { fprintf(stderr, "mfe-kbench: unknown suite %s\n", s.c_str()); return 2; }
    }

    // results: JSON for the harness, a Markdown table on stdout
    std::string js = "{\n \"tool\": \"mfe-kbench\", \"llama_commit\": \"b1b770d90d94cad9e9ea5b993bdbc383a522fb28\",\n";
    js += " \"device\": \"" + std::string(desc ? desc : "?") + "\", \"gpu_family\": \"Apple" + std::to_string(family) + "\",\n";
    js += " \"nominal_gbs\": " + std::to_string(o.nominal_gbs) + ", \"copy_gbs\": " + std::to_string(g_copy_gbs) + ",\n \"results\": [\n";
    printf("| suite | op | type | K | N | ctx | rows | copies | us/op | GB/s | %% nominal floor | %% copy floor | us/row vs 1 row |\n");
    printf("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (size_t i = 0; i < g_results.size(); ++i) {
        const Result & r = g_results[i];
        const double gbs = r.bytes / r.us / 1e3;
        const double nom = 100.0 * gbs / o.nominal_gbs;
        const double cp = g_copy_gbs > 0 ? 100.0 * gbs / g_copy_gbs : 0;
        // verify cost per row relative to the same op at 1 row
        double per_row = 0;
        for (const Result & b : g_results) {
            if (b.suite == r.suite && b.name == r.name && b.type == r.type && b.k == r.k && b.n == r.n && b.ctx == r.ctx && b.rows == 1 && r.rows > 0) {
                per_row = (r.us / r.rows) / b.us;
            }
        }
        printf("| %s | %s | %s | %d | %d | %d | %d | %d | %.1f | %.1f | %.1f | %.1f | %.3f |\n", r.suite.c_str(), r.name.c_str(),
               r.type.c_str(), r.k, r.n, r.ctx, r.rows, r.copies, r.us, gbs, nom, cp, per_row);
        char line[512];
        snprintf(line, sizeof line, "  {\"suite\": \"%s\", \"op\": \"%s\", \"type\": \"%s\", \"k\": %d, \"n\": %d, \"ctx\": %d, \"rows\": %d, "
                 "\"copies\": %d, \"us\": %.3f, \"bytes\": %.0f, \"gbs\": %.3f, \"pct_nominal\": %.2f, \"pct_copy\": %.2f, \"us_per_row_vs_1\": %.4f}%s\n",
                 r.suite.c_str(), r.name.c_str(), r.type.c_str(), r.k, r.n, r.ctx, r.rows, r.copies, r.us, r.bytes, gbs, nom, cp, per_row,
                 i + 1 < g_results.size() ? "," : "");
        js += line;
    }
    js += " ]\n}\n";
    if (!o.json.empty()) {
        FILE * f = fopen(o.json.c_str(), "w");
        if (f) { fputs(js.c_str(), f); fclose(f); }
    }
    ggml_backend_free(be);
    return 0;
}
