import Foundation
import MferenceRepackCore

/// Immutable acceptance inputs for the Maple teacher-forcing trace.
public enum MapleParityPins {
    public static let schema = "mference.maple.teacher_forcing.v1"
    public static let engine = "mference"
    public static let expectedPositions = 1_639
    public static let topK = 10
    public static let expertCacheSlots = 16
    public static let vocabularySize = 151_936
    public static let corpusSHA256 = "6a8ffabe83cf524659bb2c83137e5d41545155f4c650a83c31642c287cc95555"
    public static let corpusManifestSHA256 = "9e877b1896c6f822d75ea58cda108e336e742a93f32f74acaf3dde6099347f25"
    public static let tokenIDsSHA256 = "979f944999a0a5039bfe3a9074ca0886ea54a228663fc8ad828a8759871f261f"
    public static let tokenPolicy = "raw-utf8-lf-nfc;add_special_tokens=false;bos=false;eos=false"
    public static let topKTieBreak = "logit-desc-token-id-asc"
    public static let corpusSourceURL = "https://www.poetryfoundation.org/poems/48860/the-raven"
    public static let sourceSnapshotManifestSHA256 = "ac8b6d4b118d982b215c98697cca50ebe770ad3d8f68b7ef10a582fd52fb9a5c"
    public static let configSHA256 = "57eb521da63629196ebda2c103be929c81c1027ddf2766e7b19e2d2427f77443"
    public static let tokenizerSHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    public static let oracleEngineVersion = "0.31.3"
    public static let oracleSourceRevision = "eba96c16158f032821b0bf374ea1421cfddef0a9"
    public static let oracleRuntimeVersions = [
        "mlx": "0.32.0",
        "mlx-lm": "0.31.3",
        "numpy": "2.5.1",
        "python": "3.12.13",
        "tokenizers": "0.22.2",
        "transformers": "5.14.1",
    ]
    public static let oracleLogitDType = "float32-export-from-mlx"
    public static let oracleIntegrityPolicy = "full-sha256"
    public static let candidateLogitDType = "float32-export-from-fp16"

    /// Kept in one source of truth with the installer rather than copied here.
    public static let source = SupportedModelSource.maple

    public static var modelRevision: String { source.revision! }
    public static var sourceIndexSHA256: String { source.sourceIndexSHA256! }
}

public enum MapleParityError: Error, CustomStringConvertible, Sendable {
    case invalid(String)
    case io(String)

    public var description: String {
        switch self {
        case .invalid(let message), .io(let message): return message
        }
    }
}
