import Foundation

// GENERATED from scratch/notes-program/pins/speech-pins.json and the verified
// download (2026-11-20). Every file of every speech model the app may
// download: pinned repository revision, size and SHA-256 (Notes spec 7).
// Sources and licences: App/Resources/Speech/SOURCES.md. Owner: N2 (the
// installer); changing a pin is a contract PR.

/// One file to download, verify and install.
public struct SpeechModelFile: Sendable, Hashable {
    /// Path in the Hugging Face repository at `revision`.
    public let repoPath: String
    /// Path under `Speech/` after install. Fluid Speech Pro's joint model is
    /// installed as `JointDecisionv3.mlmodelc`, the name FluidAudio's `.v3`
    /// loader expects (spike S1).
    public let installPath: String
    public let size: Int64
    public let sha256: String
}

/// One pinned Hugging Face repository revision.
public struct SpeechModelSource: Sendable, Hashable {
    public let repo: String
    public let revision: String
    public let license: String
    public let files: [SpeechModelFile]

    /// The download URL for a file: huggingface.co only.
    public func url(for file: SpeechModelFile) -> URL {
        let path = file.repoPath.split(separator: "/").map {
            String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0)
        }.joined(separator: "/")
        return URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")!
    }
}

/// A package as Settings and the setup card show it.
public struct SpeechModelPackage: Sendable, Hashable {
    public let id: SpeechPackageID
    public let sources: [SpeechModelSource]

    public var totalBytes: Int64 { sources.reduce(0) { $0 + $1.files.reduce(0) { $0 + $1.size } } }
}

public enum SpeechModelPins {
    /// The hosts a speech download may reach, with their subdomains (the LFS
    /// redirect goes to `*.cdn.hf.co`): the same list as the GGUF downloads.
    public static let allowedHosts: [String] = ["huggingface.co", "hf.co"]

    public static func package(_ id: SpeechPackageID) -> SpeechModelPackage {
        switch id {
        case .speech: speech
        case .speakerLabels: speakerLabels
        }
    }


    public static let speech = SpeechModelPackage(id: .speech, sources: [
        SpeechModelSource(repo: "FluidVoice-Labs/fluid-speech-pro-coreml", revision: "3b9d38a93d606f84011e5a0a669b7ea146375a22", license: "CC BY 4.0", files: [
            SpeechModelFile(repoPath: "Decoder.mlmodelc/analytics/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Decoder.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "d172cb80cbf3a2cd768952557164db1b50b9c6aef16af3653d2fd7c03d9fe8e5"),
            SpeechModelFile(repoPath: "Decoder.mlmodelc/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Decoder.mlmodelc/coremldata.bin",
                            size: 491, sha256: "5059c27aa8cc44eb735247d335e18da599d15e30d67b3d2d5633679fc5806f8a"),
            SpeechModelFile(repoPath: "Decoder.mlmodelc/metadata.json", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Decoder.mlmodelc/metadata.json",
                            size: 3311, sha256: "47ee2ad3514945b6ee3ccd4171ed3b2318e38f3b929e4398ce33b50aca606083"),
            SpeechModelFile(repoPath: "Decoder.mlmodelc/model.mil", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Decoder.mlmodelc/model.mil",
                            size: 11823, sha256: "9d605291dfdaef07ef88b8d3dbf0702766f1ceea5649d5d6dee4d2322d8fa678"),
            SpeechModelFile(repoPath: "Decoder.mlmodelc/weights/weight.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Decoder.mlmodelc/weights/weight.bin",
                            size: 23604992, sha256: "7b3e3a4e3fbb580ae990613915f8ea2309226d85999ae37120912b8352d1df9e"),
            SpeechModelFile(repoPath: "Encoder.mlmodelc/analytics/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Encoder.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "ac53d09c1b1f7f00fd1a07a630d57c4193089b19fe0d3b1571a3ecf91be62aa4"),
            SpeechModelFile(repoPath: "Encoder.mlmodelc/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Encoder.mlmodelc/coremldata.bin",
                            size: 427, sha256: "abfbc09cc0b8aba7911e38390390f1c5850e3665862238035052e015659ebb74"),
            SpeechModelFile(repoPath: "Encoder.mlmodelc/metadata.json", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Encoder.mlmodelc/metadata.json",
                            size: 2758, sha256: "cc74be4442be9fe7d6428aa238364d40e91d3c0ca4ce3cfa2f4fef4ad7a42cee"),
            SpeechModelFile(repoPath: "Encoder.mlmodelc/model.mil", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Encoder.mlmodelc/model.mil",
                            size: 621903, sha256: "f71d68afe10e227c39d43af07c1aa1a88dd0eb2f63109a3e03711066e38a2e9d"),
            SpeechModelFile(repoPath: "Encoder.mlmodelc/weights/weight.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Encoder.mlmodelc/weights/weight.bin",
                            size: 613088448, sha256: "4abdc3b4af2ff9b2843105f29677f0e674d2b08638b1ce6feac60876c235c659"),
            SpeechModelFile(repoPath: "JointDecision.mlmodelc/analytics/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/JointDecisionv3.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "3dd0d66c51f69d683c587d5c88f3256d5e1690105ecf9d8ca4b0c2b8e1c883c7"),
            SpeechModelFile(repoPath: "JointDecision.mlmodelc/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/JointDecisionv3.mlmodelc/coremldata.bin",
                            size: 462, sha256: "febdb37a5ad1a119539c33cf68fac1c51f80f40f82c90b6ef312033191ff4063"),
            SpeechModelFile(repoPath: "JointDecision.mlmodelc/metadata.json", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/JointDecisionv3.mlmodelc/metadata.json",
                            size: 2728, sha256: "36fa268dd3c132c7305bbe1f17b5891a00f91caf0a883782375c35a97e2f4ada"),
            SpeechModelFile(repoPath: "JointDecision.mlmodelc/model.mil", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/JointDecisionv3.mlmodelc/model.mil",
                            size: 7178, sha256: "618bc532db01d5d80da27d0137e4b50cf2f56dde54e0686a4a1e732d74c1a55c"),
            SpeechModelFile(repoPath: "JointDecision.mlmodelc/weights/weight.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/JointDecisionv3.mlmodelc/weights/weight.bin",
                            size: 12642764, sha256: "f0939e8807dded497926a7c3634af698557ec8b155d804f8f40e367e97f90849"),
            SpeechModelFile(repoPath: "LICENSE", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/LICENSE",
                            size: 18657, sha256: "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"),
            SpeechModelFile(repoPath: "NOTICE", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/NOTICE",
                            size: 451, sha256: "fb46bca8b337792a755ec6c77e0e33ebd27d181f46540e4b88382d179942625c"),
            SpeechModelFile(repoPath: "Preprocessor.mlmodelc/analytics/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Preprocessor.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "886396da81b6b29d60d79d903bd5d1f00077a7a4323f40cd2e9214d4205d8e6b"),
            SpeechModelFile(repoPath: "Preprocessor.mlmodelc/coremldata.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Preprocessor.mlmodelc/coremldata.bin",
                            size: 429, sha256: "38f130eac9ffde8c684043d76e3eecdd3b6f9564d9efa59f7a87a34a272a4901"),
            SpeechModelFile(repoPath: "Preprocessor.mlmodelc/metadata.json", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Preprocessor.mlmodelc/metadata.json",
                            size: 2606, sha256: "c25d98c19a181b7fa65ee50c34f5066f67172c22f0bf116a42b59443ee4bc37d"),
            SpeechModelFile(repoPath: "Preprocessor.mlmodelc/model.mil", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Preprocessor.mlmodelc/model.mil",
                            size: 11891, sha256: "71d4ad016334b3ad73ea7b69b6813feefe3b81f59e0b5e924f90f088cb02a7c9"),
            SpeechModelFile(repoPath: "Preprocessor.mlmodelc/weights/weight.bin", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/Preprocessor.mlmodelc/weights/weight.bin",
                            size: 2150580, sha256: "44180071e400951080786feb46452bffc4fb26363a703655ebadb760c8ad4030"),
            SpeechModelFile(repoPath: "config.json", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/config.json",
                            size: 646, sha256: "b688c4d174c4fb043b368dad402ae91d4ae68cf2a53db42ff663340913c907db"),
            SpeechModelFile(repoPath: "parakeet_vocab.json", installPath: "fluid-speech-pro-coreml/3b9d38a93d606f84011e5a0a669b7ea146375a22/parakeet_vocab.json",
                            size: 151122, sha256: "7ec60e05f1b24480736ec0eed40900f4626bce1fa9a60fd700ec7e2a59198735"),
        ]),
        SpeechModelSource(repo: "FluidInference/silero-vad-coreml", revision: "b419383c55c110e2c9271fa6ee0ea83d03c70d96", license: "MIT", files: [
            SpeechModelFile(repoPath: "silero-vad-unified-256ms-v6.2.1.mlmodelc/analytics/coremldata.bin", installPath: "silero-vad-coreml/b419383c55c110e2c9271fa6ee0ea83d03c70d96/silero-vad-unified-256ms-v6.2.1.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "8067594eb3126ab8318af507f0c00cabfed40d5fedb8a0ee5075dd02e903d909"),
            SpeechModelFile(repoPath: "silero-vad-unified-256ms-v6.2.1.mlmodelc/coremldata.bin", installPath: "silero-vad-coreml/b419383c55c110e2c9271fa6ee0ea83d03c70d96/silero-vad-unified-256ms-v6.2.1.mlmodelc/coremldata.bin",
                            size: 625, sha256: "7db35a4fd995222a7fb0129713473b15d1462572ab4a2e5e4d56bcaad9e40f41"),
            SpeechModelFile(repoPath: "silero-vad-unified-256ms-v6.2.1.mlmodelc/metadata.json", installPath: "silero-vad-coreml/b419383c55c110e2c9271fa6ee0ea83d03c70d96/silero-vad-unified-256ms-v6.2.1.mlmodelc/metadata.json",
                            size: 3335, sha256: "2740be542c611e1ba358e1849b4e265c65cdf0b17192767e1e5de86a31ac94d6"),
            SpeechModelFile(repoPath: "silero-vad-unified-256ms-v6.2.1.mlmodelc/model.mil", installPath: "silero-vad-coreml/b419383c55c110e2c9271fa6ee0ea83d03c70d96/silero-vad-unified-256ms-v6.2.1.mlmodelc/model.mil",
                            size: 176918, sha256: "c6a9d1bf22d413265da0a07a1d14151c3ea2fad296b3aa5859275b33ef1c3270"),
            SpeechModelFile(repoPath: "silero-vad-unified-256ms-v6.2.1.mlmodelc/weights/weight.bin", installPath: "silero-vad-coreml/b419383c55c110e2c9271fa6ee0ea83d03c70d96/silero-vad-unified-256ms-v6.2.1.mlmodelc/weights/weight.bin",
                            size: 882304, sha256: "53ecc8b5081146140ab654c89109cf001f2183abddd7a2411c5081feeffff063"),
        ]),
    ])

    public static let speakerLabels = SpeechModelPackage(id: .speakerLabels, sources: [
        SpeechModelSource(repo: "FluidInference/speaker-diarization-coreml", revision: "7eab74f3463c736418d0eb10296261ae581146c9", license: "CC BY 4.0 (Community-1 set, NOTICE.md)", files: [
            SpeechModelFile(repoPath: "Embedding.mlmodelc/analytics/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Embedding.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "8d6706436639b53830b4dbe8aaf9c9a843f7f582d63e16f3cb8bb7c6ccd58682"),
            SpeechModelFile(repoPath: "Embedding.mlmodelc/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Embedding.mlmodelc/coremldata.bin",
                            size: 704, sha256: "4a705bac27d151d9642f37609296042a15602a42253039e0921dc9e75da7e004"),
            SpeechModelFile(repoPath: "Embedding.mlmodelc/metadata.json", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Embedding.mlmodelc/metadata.json",
                            size: 2818, sha256: "1854371eb6b438fb8aeac96afb45c999af7902581c06afdfcd7ff3cb1ce66be5"),
            SpeechModelFile(repoPath: "Embedding.mlmodelc/model.mil", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Embedding.mlmodelc/model.mil",
                            size: 78432, sha256: "22fa958aef72a561c21f874a07cbdcd30fdf40ee961c0bc2fb67c119273b46d3"),
            SpeechModelFile(repoPath: "Embedding.mlmodelc/weights/weight.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Embedding.mlmodelc/weights/weight.bin",
                            size: 13412288, sha256: "99356b2985b8d43880a657024d941d450b38820451ccff903f76ed4e52d1868b"),
            SpeechModelFile(repoPath: "FBank.mlmodelc/analytics/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/FBank.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "0e8bd3a8b82ac123580989f490e4d9245127c535857630b543311268accc3f0a"),
            SpeechModelFile(repoPath: "FBank.mlmodelc/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/FBank.mlmodelc/coremldata.bin",
                            size: 853, sha256: "57ac436bb0671cbb5527a339134d695f752eb77f7a18966b93c6835335595759"),
            SpeechModelFile(repoPath: "FBank.mlmodelc/metadata.json", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/FBank.mlmodelc/metadata.json",
                            size: 3409, sha256: "2623785f5d186893b82d01e84aa33a7704ef763c3309e02055f22dc9d871ce9a"),
            SpeechModelFile(repoPath: "FBank.mlmodelc/model.mil", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/FBank.mlmodelc/model.mil",
                            size: 15667, sha256: "27aaeb21569e81bdbe2eef87789f50a37cfea800039bd134448a9417de2f30ed"),
            SpeechModelFile(repoPath: "FBank.mlmodelc/weights/weight.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/FBank.mlmodelc/weights/weight.bin",
                            size: 1776896, sha256: "9e83fdd3ea78064b078069e4d9141603c61c47a27fd19e7e3142ff7476f8db36"),
            SpeechModelFile(repoPath: "LICENSE", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/LICENSE",
                            size: 19759, sha256: "736763bf08829e36266b8a3bebcc041ac68575424e976490671a7e6b43a525cf"),
            SpeechModelFile(repoPath: "NOTICE.md", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/NOTICE.md",
                            size: 4247, sha256: "d8e53d399c5345cabc4763732ba096e7e18ff42c861d57b73edb73b0c9ce9356"),
            SpeechModelFile(repoPath: "PLDA.mlmodelc/analytics/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PLDA.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "8e862420707c6fc86c5720adcbc811286235970cf5e0c5afdb7e10e03c937026"),
            SpeechModelFile(repoPath: "PLDA.mlmodelc/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PLDA.mlmodelc/coremldata.bin",
                            size: 535, sha256: "3d5ddb4cf367ad23fb512587f751cddbe4cbf867f53c5395435e6719a6acfd20"),
            SpeechModelFile(repoPath: "PLDA.mlmodelc/metadata.json", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PLDA.mlmodelc/metadata.json",
                            size: 1975, sha256: "95eab9a4c9ceb9b16bd36d3eee5530796a0cf78153547088c6c40e9e43999b98"),
            SpeechModelFile(repoPath: "PLDA.mlmodelc/model.mil", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PLDA.mlmodelc/model.mil",
                            size: 5088, sha256: "0c61b5f3fc8d482218ecf3f213ff42e40d6fc38265d3dcf5065da113974b25b8"),
            SpeechModelFile(repoPath: "PLDA.mlmodelc/weights/weight.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PLDA.mlmodelc/weights/weight.bin",
                            size: 199616, sha256: "566c14f27af4ef1a4bdfb8ea875adeedd7026a85e1026659f48a3d305d51de0c"),
            SpeechModelFile(repoPath: "PldaRho.mlmodelc/analytics/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PldaRho.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "8940ea6044dbcbefa22da8cc41e0b485e1fb5ed89aecaf37c6e0c483a97ddcd7"),
            SpeechModelFile(repoPath: "PldaRho.mlmodelc/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PldaRho.mlmodelc/coremldata.bin",
                            size: 763, sha256: "4d9741477f721c79b09fcdfe455110c4b7d4272e2de3496bf1729d966d3ee418"),
            SpeechModelFile(repoPath: "PldaRho.mlmodelc/metadata.json", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PldaRho.mlmodelc/metadata.json",
                            size: 2749, sha256: "b314cf25a93e46b4076883a6f5a2f8848b73c3851bd9d36074d067f35a1c7945"),
            SpeechModelFile(repoPath: "PldaRho.mlmodelc/model.mil", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PldaRho.mlmodelc/model.mil",
                            size: 7613, sha256: "83aee2e5310d19b5f202aea97d07a0e12102556d1b32ef3ed08b36f7f9725041"),
            SpeechModelFile(repoPath: "PldaRho.mlmodelc/weights/weight.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/PldaRho.mlmodelc/weights/weight.bin",
                            size: 200192, sha256: "80f7d229202636d372428c90596f11a91545f07da77259f07153aaf225914a36"),
            SpeechModelFile(repoPath: "Segmentation.mlmodelc/analytics/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Segmentation.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "64265f8e7ad41a5f68d630c15288c2499cca5892ad49e20096819cdeac004cdb"),
            SpeechModelFile(repoPath: "Segmentation.mlmodelc/coremldata.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Segmentation.mlmodelc/coremldata.bin",
                            size: 812, sha256: "ea51481b8bd3e496ad3cf16f066ddaa37f20e8772eaac76b3393c28de20e06bc"),
            SpeechModelFile(repoPath: "Segmentation.mlmodelc/metadata.json", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Segmentation.mlmodelc/metadata.json",
                            size: 3410, sha256: "88dbf0b07208fe142e1729c2b4c974ad3599fcb2ae5d5f18fce782b225384124"),
            SpeechModelFile(repoPath: "Segmentation.mlmodelc/model.mil", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Segmentation.mlmodelc/model.mil",
                            size: 43063, sha256: "d37e4ce30b406a6b34f765f769b9baed3178cc0c2b2e299c641daa43a052dd3f"),
            SpeechModelFile(repoPath: "Segmentation.mlmodelc/weights/weight.bin", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/Segmentation.mlmodelc/weights/weight.bin",
                            size: 5959360, sha256: "c3189a64946c75bc24fcb98afe89ad78c52bdbadfdf65e857fb1b81e2cc9fbb2"),
            SpeechModelFile(repoPath: "plda-parameters.json", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/plda-parameters.json",
                            size: 89416, sha256: "38ee28d4269c076cef254ee760bbd811f0738a92e0f01f9699ad372828c5de8f"),
            SpeechModelFile(repoPath: "xvector-transform.json", installPath: "speaker-diarization-coreml/7eab74f3463c736418d0eb10296261ae581146c9/xvector-transform.json",
                            size: 177499, sha256: "f7cd5cc16e63e2d89db052a23018ecfc47a311998ed1e9e39838fbac65048688"),
        ]),
        SpeechModelSource(repo: "FluidInference/ls-eend-coreml", revision: "28ce1b1f8ef186729df63b3886fbaae7bc10c4a1", license: "MIT", files: [
            SpeechModelFile(repoPath: "AMI/ls_eend_ami_step.json", installPath: "ls-eend-coreml/28ce1b1f8ef186729df63b3886fbaae7bc10c4a1/AMI/ls_eend_ami_step.json",
                            size: 1237, sha256: "3eab7020357a5639a8a5dc6631e85bfb14ec118035367f7eb60beaaae7727d90"),
            SpeechModelFile(repoPath: "AMI/ls_eend_ami_step.mlmodelc/analytics/coremldata.bin", installPath: "ls-eend-coreml/28ce1b1f8ef186729df63b3886fbaae7bc10c4a1/AMI/ls_eend_ami_step.mlmodelc/analytics/coremldata.bin",
                            size: 243, sha256: "6c29e4e605586aca4f45ad01254b2fe365180fb9a42ac6c2fa65f114cc603c27"),
            SpeechModelFile(repoPath: "AMI/ls_eend_ami_step.mlmodelc/coremldata.bin", installPath: "ls-eend-coreml/28ce1b1f8ef186729df63b3886fbaae7bc10c4a1/AMI/ls_eend_ami_step.mlmodelc/coremldata.bin",
                            size: 742, sha256: "e88c8cc32b10c236622478631ba236544a578cab395119d1db21efc7e9de08b0"),
            SpeechModelFile(repoPath: "AMI/ls_eend_ami_step.mlmodelc/metadata.json", installPath: "ls-eend-coreml/28ce1b1f8ef186729df63b3886fbaae7bc10c4a1/AMI/ls_eend_ami_step.mlmodelc/metadata.json",
                            size: 6498, sha256: "0b32d25fa92cff80d2dee9cde27d31860c6baa428a0711f4d91d787fb7f6c0a3"),
            SpeechModelFile(repoPath: "AMI/ls_eend_ami_step.mlmodelc/model.mil", installPath: "ls-eend-coreml/28ce1b1f8ef186729df63b3886fbaae7bc10c4a1/AMI/ls_eend_ami_step.mlmodelc/model.mil",
                            size: 178602, sha256: "cdbe7ec850c0932df150c75eceeaae4ec4c3bd218c581aff81a62cf9cf65a16d"),
            SpeechModelFile(repoPath: "AMI/ls_eend_ami_step.mlmodelc/weights/weight.bin", installPath: "ls-eend-coreml/28ce1b1f8ef186729df63b3886fbaae7bc10c4a1/AMI/ls_eend_ami_step.mlmodelc/weights/weight.bin",
                            size: 44388992, sha256: "56e27813e10448eb4454b17574becc305015ad657ce6cd3896044b7c8b95bcc3"),
            SpeechModelFile(repoPath: "LICENSE", installPath: "ls-eend-coreml/28ce1b1f8ef186729df63b3886fbaae7bc10c4a1/LICENSE",
                            size: 1072, sha256: "bcd00ee53d35b9a089a115fdb8eb6d8ea21d4161deb95c5cb9f91168fd0c7a33"),
        ]),
    ])
}
