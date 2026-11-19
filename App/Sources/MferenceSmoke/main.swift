import Darwin
import Foundation
import IOKit.ps
import MferenceAppCore

// End-to-end smoke for one model through the app's own engine path
// (`EngineService.live()`, in-process, as the app runs it). One request,
// then shutdown. Run it only under the machine lock, after
// `lock.sh preflight`, on AC power, one model at a time:
//
//   lock.sh run a2 "app smoke quick" -- App/.build/release/MferenceSmoke --model quick \
//       --out /Users/.../scratch/app-program/results/smoke
//
// C-A2 (web answers): add `--web`. The question (default
// `WebCheckpoint.defaultQuestion`, or `--prompt`) goes through the chat with
// web search on (bundled SearXNG, which `MFERENCE_SEARXNG_DIR` or
// App/Resources/SearXNG/runtime provides); it passes when the model searched
// and the answer cites sources. The result's `web` field has the details.
//
// Warm decode: add `--warmup` to send a short request on the loaded model
// first, so the main request's decode is warm; `--temperature 0` is greedy,
// as the engine's own speed runs use (default: the checkpoint's sampling).
// Each result records decode tok/s.
//
// Exit 0 pass, 1 the request failed, 2 bad arguments, 4 swap grew during
// the run (stop all model runs and report), 5 not on AC power at the start
// (nothing loaded) or at the end, 6 MFERENCE_APP_SMOKE=1 not set (nothing
// loaded), 8 swap could not be read.

// A model run: refuse unless the caller says it is one on purpose.
guard ProcessInfo.processInfo.environment["MFERENCE_APP_SMOKE"] == "1" else {
    FileHandle.standardError.write(Data("MferenceSmoke: loads a model; run it only under lock.sh run with MFERENCE_APP_SMOKE=1\n".utf8))
    exit(6)
}

// Must run before anything creates a Metal device (spec 4.3.2).
setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 1)

struct SmokeResult: Codable {
    /// The commit the binary was built from (`--commit`), for the record.
    var commit: String?
    var model: String
    var modelID: String
    var engine: String
    var location: String?
    var sizeBytes: UInt64
    var startedUTC: String
    var prompt: String
    var answer: String = ""
    var reasoningCharacters: Int = 0
    var finish: String?
    var promptTokens: Int?
    var completionTokens: Int?
    var loadSeconds: Double?
    var prefillSeconds: Double?
    var decodeSeconds: Double?
    var wallSeconds: Double = 0
    var swapUsedBeforeBytes: UInt64
    var swapUsedAfterBytes: UInt64 = 0
    var swapGrewBytes: Int64 = 0
    var powerSourceAtEnd: String?
    /// `--web` runs: what the web answer searched, read and cited.
    var web: WebCheckpointOutcome?
    /// `--warmup`: the short request sent first on the loaded model.
    var warmupCompletionTokens: Int?
    var warmupPrefillSeconds: Double?
    var warmupDecodeSeconds: Double?
    /// completionTokens / decodeSeconds of the main request.
    var decodeTokensPerSecond: Double?
    /// Nil: the checkpoint's sampling defaults; 0 is greedy.
    var temperature: Float?
    var error: String?
    var passed = false
}

/// Swap in use, or nil when the kernel does not say.
func swapUsedBytes() -> UInt64? {
    var usage = xsw_usage()
    var size = MemoryLayout<xsw_usage>.size
    guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return nil }
    return usage.xsu_used
}

func powerSource() -> String {
    guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
          let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() else { return "unknown" }
    return type as String
}

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write(Data("MferenceSmoke: \(message)\n".utf8))
    exit(code)
}

var arguments = CommandLine.arguments.dropFirst()
var modelName: String?
var outDirectory: URL?
var prompt = "Answer in one short sentence: what is the capital of France?"
var maxTokens = 64
var commit: String?
var webMode = false
var warmup = false
var temperature: Float?
var promptGiven = false
while let argument = arguments.popFirst() {
    switch argument {
    case "--model": modelName = arguments.popFirst()
    case "--out": outDirectory = arguments.popFirst().map { URL(fileURLWithPath: $0, isDirectory: true) }
    case "--prompt":
        prompt = arguments.popFirst() ?? prompt
        promptGiven = true
    case "--web": webMode = true
    case "--warmup": warmup = true
    case "--temperature": temperature = arguments.popFirst().flatMap(Float.init)
    case "--max-tokens": maxTokens = arguments.popFirst().flatMap(Int.init) ?? maxTokens
    case "--commit": commit = arguments.popFirst()
    default: fail("unknown argument \(argument)", code: 2)
    }
}
guard let modelName, let model = AppModel(rawValue: modelName) else {
    fail("--model must be one of \(AppModel.allCases.map(\.rawValue).joined(separator: ", "))", code: 2)
}
if webMode, !promptGiven { prompt = WebCheckpoint.defaultQuestion }

let power = powerSource()
guard power == kIOPMACPowerKey else { fail("power source is '\(power)'; smoke runs need AC power", code: 5) }

let started = ContinuousClock.now
let startedUTC = ISO8601DateFormatter().string(from: Date())
guard let swapBefore = swapUsedBytes() else { fail("cannot read vm.swapusage", code: 8) }
let idle = IdleUnloadControl(delay: nil)
let engine = EngineService.live(idleUnload: idle)
let option = await engine.models().first { $0.model == model }
var result = SmokeResult(commit: commit, model: model.name, modelID: model.modelID, engine: model.engineLabel,
                         location: option?.location?.path, sizeBytes: option?.sizeBytes ?? 0,
                         startedUTC: startedUTC, prompt: prompt, swapUsedBeforeBytes: swapBefore)

result.temperature = temperature

do {
    if case .notInstalled(let reason)? = option?.availability {
        throw EngineServiceError.notInstalled(model: model.name, reason: reason)
    }
    let loadStarted = ContinuousClock.now
    try await engine.load(model)
    let loadTime = loadStarted.duration(to: .now)
    result.loadSeconds = Double(loadTime.components.seconds) + Double(loadTime.components.attoseconds) / 1e18
    if webMode {
        // In-memory settings: the run turns web search on (and offline mode
        // off) without touching the person's own settings.
        let settings = AppSettings(store: InMemorySettingsStore())
        settings.webSearchEnabled = true
        settings.offlineMode = false
        let web = WebAnswers(settings: settings, legacyStore: InMemorySettingsStore())
        guard web.process != nil else {
            throw WebError.searchUnavailable("no SearXNG runtime; run App/Scripts/searxng/prepare.sh --offline")
        }
        let outcome = await WebCheckpoint.run(question: prompt, model: model, service: engine, tools: web)
        await web.shutdown()
        result.web = outcome
        result.answer = outcome.answer
        result.reasoningCharacters = outcome.reasoningCharacters
        result.finish = outcome.finish
        result.promptTokens = outcome.promptTokens
        result.completionTokens = outcome.completionTokens
        result.passed = outcome.passed
        if !outcome.passed { result.error = outcome.error ?? "web answer failed" }
    } else {
        if warmup {
            let short = GenerationOptions(thinking: model.supportsThinking ? false : nil,
                                          maximumCompletionTokens: 16, temperature: temperature)
            for try await update in engine.generate(model: model,
                                                    turns: [ChatTurn(role: .user, content: "Say hello.")],
                                                    options: short) {
                if case .completed(let summary) = update {
                    result.warmupCompletionTokens = summary.completionTokens
                    result.warmupPrefillSeconds = summary.prefillSeconds
                    result.warmupDecodeSeconds = summary.decodeSeconds
                }
            }
        }
        let options = GenerationOptions(thinking: model.supportsThinking ? false : nil,
                                        maximumCompletionTokens: maxTokens, temperature: temperature)
        for try await update in engine.generate(model: model, turns: [ChatTurn(role: .user, content: prompt)],
                                                options: options) {
            switch update {
            case .content(let text): result.answer += text
            case .reasoning(let text): result.reasoningCharacters += text.count
            case .completed(let summary):
                result.finish = summary.finish.rawValue
                result.promptTokens = summary.promptTokens
                result.completionTokens = summary.completionTokens
                result.prefillSeconds = summary.prefillSeconds
                result.decodeSeconds = summary.decodeSeconds
                if let seconds = summary.decodeSeconds, seconds > 0 {
                    result.decodeTokensPerSecond = Double(summary.completionTokens) / seconds
                }
            case .prepared, .toolCall: break
            }
        }
        let answer = result.answer.trimmingCharacters(in: .whitespacesAndNewlines)
        result.passed = result.finish != nil && !answer.isEmpty
        if !result.passed { result.error = "empty answer" }
    }
} catch {
    result.error = (error as? EngineServiceError)?.userMessage ?? String(describing: error)
}
await engine.shutdown()

let wall = started.duration(to: .now)
result.wallSeconds = Double(wall.components.seconds) + Double(wall.components.attoseconds) / 1e18
let swapAfter = swapUsedBytes()
result.swapUsedAfterBytes = swapAfter ?? 0
result.swapGrewBytes = swapAfter.map { Int64($0) - Int64(swapBefore) } ?? 0
let powerAtEnd = powerSource()
result.powerSourceAtEnd = powerAtEnd
if swapAfter == nil { result.passed = false; result.error = "cannot read vm.swapusage at the end" }
if powerAtEnd != kIOPMACPowerKey { result.passed = false; result.error = "power source changed to '\(powerAtEnd)'" }

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let json = (try? encoder.encode(result)) ?? Data()
print(String(decoding: json, as: UTF8.self))
if let outDirectory {
    try? FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)
    let stamp = startedUTC.replacingOccurrences(of: ":", with: "")
    try? json.write(to: outDirectory.appending(path: "\(model.rawValue)\(webMode ? "-web" : "")-\(stamp).json"))
}

if swapAfter == nil { fail("cannot read vm.swapusage at the end", code: 8) }
if powerAtEnd != kIOPMACPowerKey { fail("power source changed to '\(powerAtEnd)' during the run", code: 5) }
if result.swapGrewBytes > 0 {
    fail("swap grew by \(result.swapGrewBytes) bytes during the run: stop all model runs and report", code: 4)
}
exit(result.passed ? 0 : 1)
