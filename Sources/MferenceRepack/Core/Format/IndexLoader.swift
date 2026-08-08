import CoreFoundation
import Foundation

/// Parses `model.safetensors.index.json` and `config.json -> quantization`.
enum IndexLoader {

    struct SourceMetadata {
        let indexPath: String
        let configPath: String
        let indexSha256Hex: String
        /// `tensor_name -> shard_filename`
        let weightMap: [String: String]
        /// Base bits / group_size / mode for any tensor not in the override table.
        let baseBits: Int
        let baseGroupSize: Int
        let baseMode: String
        /// Per-tensor overrides (keyed by tensor name **without** the trailing
        /// `.weight` — matches the way `config.json` writes them).
        let bitsOverrides: [String: QuantSpec]
        /// Resolved set of shard files referenced by the index, in
        /// encounter order. Order is stable enough for sequential I/O.
        let shardFilenames: [String]
    }

    static func load(snapshotDir: String) throws -> SourceMetadata {
        let indexPath  = (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json")
        let configPath = (snapshotDir as NSString).appendingPathComponent("config.json")

        let weightMap: [String: String]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let m = root["weight_map"] as? [String: String] else {
                throw RepackError.indexJsonInvalid(path: indexPath, detail: "no weight_map")
            }
            weightMap = m
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "\(error)")
        }

        let indexSha = try Sha256Stream.hashFile(path: indexPath)

        var baseBits = 4
        var baseGroup = 64
        var baseMode = "affine"
        var overrides: [String: QuantSpec] = [:]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
            }
            guard let quant = root["quantization"] as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "no quantization slot")
            }
            let isMaple = (root["model_type"] as? String) == "maple"
            if isMaple {
                guard exactInteger(quant["bits"]) == 2,
                      exactInteger(quant["group_size"]) == 128,
                      quant["mode"] as? String == "affine" else {
                    throw RepackError.configJsonInvalid(
                        path: configPath,
                        detail: "Maple requires INT2/group-128 affine base quantization")
                }
                let overrideKeys = Set(quant.keys).subtracting(["bits", "group_size", "mode"])
                let expectedOverrideKeys: Set<String> = ["model.word_embeddings", "lm_head"]
                guard overrideKeys == expectedOverrideKeys else {
                    throw RepackError.configJsonInvalid(
                        path: configPath,
                        detail: "Maple requires only model.word_embeddings and lm_head quantization overrides")
                }
                for key in expectedOverrideKeys {
                    guard let override = quant[key] as? [String: Any],
                          exactInteger(override["bits"]) == 4,
                          exactInteger(override["group_size"]) == 64,
                          override.keys.allSatisfy({ $0 == "bits" || $0 == "group_size" }) else {
                        throw RepackError.configJsonInvalid(
                            path: configPath,
                            detail: "Maple requires \(key) INT4/group-64 quantization")
                    }
                }
                baseBits = 2
                baseGroup = 128
                baseMode = "affine"
                overrides = Dictionary(uniqueKeysWithValues: expectedOverrideKeys.map {
                    ($0, QuantSpec(bits: 4, groupSize: 64))
                })
            } else {
                if let b = quant["bits"] as? Int      { baseBits  = b }
                if let g = quant["group_size"] as? Int { baseGroup = g }
                if let m = quant["mode"] as? String   { baseMode  = m }
                for (k, v) in quant where !(k == "bits" || k == "group_size" || k == "mode") {
                    guard let entry = v as? [String: Any] else { continue }
                    let bits = (entry["bits"] as? Int) ?? baseBits
                    let g    = (entry["group_size"] as? Int) ?? baseGroup
                    // Resident-tensor kernels assume the base group size;
                    // only 2-bit streamed routed experts support a deviating
                    // group (DeepSeek V4 ships gate_proj at group 32).
                    guard g == baseGroup || (bits == 2 && (g == 32 || g == 64)) else {
                        throw RepackError.configJsonInvalid(
                            path: configPath,
                            detail: "quantization override \(k) group_size \(g) != base \(baseGroup)")
                    }
                    overrides[k] = QuantSpec(bits: bits, groupSize: g)
                }
            }
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.configJsonInvalid(path: configPath, detail: "\(error)")
        }

        var seen = Set<String>()
        var shards: [String] = []
        for k in weightMap.keys.sorted() {
            let shard = weightMap[k]!
            if !seen.contains(shard) { seen.insert(shard); shards.append(shard) }
        }

        return SourceMetadata(indexPath: indexPath, configPath: configPath,
                              indexSha256Hex: indexSha,
                              weightMap: weightMap,
                              baseBits: baseBits, baseGroupSize: baseGroup,
                              baseMode: baseMode,
                              bitsOverrides: overrides,
                              shardFilenames: shards)
    }

    /// Resolves the bits/group for one tensor name (with or without `.weight`).
    static func quantSpec(forTensor name: String,
                                 meta: SourceMetadata) -> QuantSpec {
        let stripped = name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
        if let o = meta.bitsOverrides[stripped] { return o }
        return QuantSpec(bits: meta.baseBits, groupSize: meta.baseGroupSize)
    }

    private static func exactInteger(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              NSNumber(value: number.int64Value).compare(number) == .orderedSame else {
            return nil
        }
        return Int(exactly: number.int64Value)
    }
}
