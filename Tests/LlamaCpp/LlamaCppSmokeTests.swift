import Foundation
import LlamaCpp
import Testing

/// Model-free checks that the vendored llama.cpp target builds, links and runs:
/// the backend registry, the Metal kernels shipped as a resource, the GGUF
/// writer and reader, and libllama's error path for a missing model.
@Suite("LlamaCpp smoke", .serialized)
struct LlamaCppSmokeTests {
    init() {
        llama_backend_init()
    }

    @Test func reportsPinnedVersion() {
        #expect(String(cString: ggml_version()) == "1.0.0")
        #expect(String(cString: ggml_commit()) == "b1b770d9")
        #expect(llama_supports_mmap())
        #expect(llama_supports_gpu_offload())
        #expect(!String(cString: llama_print_system_info()).isEmpty)
    }

    @Test func registersCPUAndMetalBackends() {
        let names = (0..<ggml_backend_reg_count()).map {
            String(cString: ggml_backend_reg_name(ggml_backend_reg_get($0)))
        }
        #expect(names.contains("CPU"))
        #expect(names.contains("MTL"))
        #expect(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nil)
    }

    /// Runs `add(mul_mat(a, b), c)` on the Metal and CPU backends and compares
    /// both with a Swift reference. Initialising the Metal backend compiles
    /// every kernel library from the bundled sources, so a missing or
    /// misplaced resource fails here.
    @Test func metalAndCPUMatchReference() throws {
        let k = 64, m = 8, n = 4
        var rng = SplitMix64(seed: 0x6c6c_616d_6163_7070)
        let a = (0..<(k * m)).map { _ in rng.nextFloat() }
        let b = (0..<(k * n)).map { _ in rng.nextFloat() }
        let c = (0..<(m * n)).map { _ in rng.nextFloat() }

        var reference = [Float](repeating: 0, count: m * n)
        for col in 0..<n {
            for row in 0..<m {
                var sum: Float = 0
                for i in 0..<k { sum += a[i + row * k] * b[i + col * k] }
                reference[row + col * m] = sum + c[row + col * m]
            }
        }

        let metal = try #require(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nil))
        defer { ggml_backend_free(metal) }
        #expect(String(cString: ggml_backend_name(metal)).hasPrefix("MTL"))
        let cpu = try #require(ggml_backend_cpu_init())
        defer { ggml_backend_free(cpu) }

        for backend in [metal, cpu] {
            // ggml_backend_graph_compute aborts the process on an op the
            // backend cannot run; a paravirtual CI GPU may lack what MUL_MAT
            // needs. Then only the kernel build above is checked for Metal.
            guard let out = try Self.compute(backend: backend, k: k, m: m, n: n, a: a, b: b, c: c) else {
                print("LlamaCpp smoke: \(String(cString: ggml_backend_name(backend))) does not support every op; compute skipped")
                continue
            }
            for (got, want) in zip(out, reference) {
                #expect(abs(got - want) <= 1e-4 * max(1, abs(want)))
            }
        }
    }

    @Test func ggufRoundTripsMetadata() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("llamacpp-smoke-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try #require(gguf_init_empty())
        gguf_set_val_u32(writer, "mference.smoke", 32_768)
        gguf_set_val_str(writer, "general.architecture", "qwen35")
        #expect(gguf_write_to_file(writer, url.path, false))
        gguf_free(writer)

        let reader = try #require(gguf_init_from_file(url.path, gguf_init_params(no_alloc: true, ctx: nil)))
        defer { gguf_free(reader) }
        let key = gguf_find_key(reader, "mference.smoke")
        #expect(key >= 0)
        #expect(gguf_get_val_u32(reader, key) == 32_768)
        let arch = gguf_find_key(reader, "general.architecture")
        #expect(String(cString: gguf_get_val_str(reader, arch)) == "qwen35")
    }

    @Test func missingModelFailsCleanly() {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("llamacpp-smoke-missing-\(UUID().uuidString).gguf").path
        let model = llama_model_load_from_file(path, llama_model_default_params())
        #expect(model == nil)
    }

    private static func compute(
        backend: ggml_backend_t, k: Int, m: Int, n: Int, a: [Float], b: [Float], c: [Float]
    ) throws -> [Float]? {
        let params = ggml_init_params(
            mem_size: 8 * ggml_tensor_overhead() + ggml_graph_overhead(),
            mem_buffer: nil,
            no_alloc: true
        )
        let ctx = try #require(ggml_init(params))
        defer { ggml_free(ctx) }

        let ta = try #require(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Int64(k), Int64(m)))
        let tb = try #require(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Int64(k), Int64(n)))
        let tc = try #require(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Int64(m), Int64(n)))
        let result = try #require(ggml_add(ctx, ggml_mul_mat(ctx, ta, tb), tc))
        let graph = try #require(ggml_new_graph(ctx))
        ggml_build_forward_expand(graph, result)
        for i in 0..<ggml_graph_n_nodes(graph) where !ggml_backend_supports_op(backend, ggml_graph_node(graph, i)) {
            return nil
        }

        let buffer = try #require(ggml_backend_alloc_ctx_tensors(ctx, backend))
        defer { ggml_backend_buffer_free(buffer) }
        a.withUnsafeBytes { ggml_backend_tensor_set(ta, $0.baseAddress, 0, $0.count) }
        b.withUnsafeBytes { ggml_backend_tensor_set(tb, $0.baseAddress, 0, $0.count) }
        c.withUnsafeBytes { ggml_backend_tensor_set(tc, $0.baseAddress, 0, $0.count) }

        #expect(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS)

        var out = [Float](repeating: 0, count: m * n)
        out.withUnsafeMutableBytes { ggml_backend_tensor_get(result, $0.baseAddress, 0, $0.count) }
        return out
    }
}

/// Deterministic inputs in [-1, 1) without a test-support dependency.
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat() -> Float {
        Float(next() >> 40) / Float(1 << 24) * 2 - 1
    }
}
