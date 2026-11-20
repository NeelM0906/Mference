import Foundation

/// The app's speech service until the live `SpeechService` (team N2) is wired
/// into `AppServices.live`: nothing is installed and every action fails
/// plainly. It never downloads, records or loads anything.
public final class UnavailableSpeechService: SpeechServicing {
    private static let reason = SpeechError.failed("The speech service is not built yet.")

    public init() {}

    public var status: SpeechStatus { SpeechStatus() }

    public func statusUpdates() -> AsyncStream<SpeechStatus> {
        AsyncStream { continuation in
            continuation.yield(SpeechStatus())
            continuation.finish()
        }
    }

    public func downloadSize(of package: SpeechPackageID) -> Int64 { 0 }
    public func install(_ package: SpeechPackageID) async throws { throw Self.reason }
    public func pauseInstall(_ package: SpeechPackageID) async {}
    public func cancelInstall(_ package: SpeechPackageID) async {}
    public func remove(_ package: SpeechPackageID) async throws {}
    public func startDictation(_ options: DictationOptions) async throws -> DictationSession {
        throw SpeechError.notInstalled(.speech)
    }
    public func startMeeting(_ options: MeetingOptions) async throws -> MeetingRecording {
        throw SpeechError.notInstalled(.speech)
    }
    public func transcribeFile(_ url: URL, options: FileTranscriptionOptions) async throws -> FileJob {
        throw SpeechError.notInstalled(.speech)
    }
    public func cancel(_ job: FileJob.ID) async {}
    public func inputDevices() async -> [AudioInputDevice] { [] }
    public func runningAudioApps() async -> [RunningAudioApp] { [] }
    public func recordingsSize() async -> Int64 { 0 }
    public func deleteRecordings(for noteID: UUID) async throws {}
    public func recordingsDirectory(for noteID: UUID) -> URL {
        AppStorageLocation.applicationSupport.appending(path: "Recordings/\(noteID.uuidString)", directoryHint: .isDirectory)
    }
    public func recoverInterruptedRecordings() async -> [InterruptedRecording] { [] }
    public func shutdown() async {}
}
