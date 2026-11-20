import MferenceAppCore
import MferenceDesign
import SwiftUI
import UniformTypeIdentifiers

/// One row of the file queue, as the sheet shows it.
public struct FileQueueRow: Sendable, Hashable, Identifiable {
    public enum Phase: Sendable, Hashable {
        case copying, queued, running, paused, done, failed(String), cancelled
    }

    public var id: UUID
    public var noteID: UUID
    public var fileName: String
    public var duration: TimeInterval?
    public var languageCode: String?
    /// Speakers found (Done rows with speaker labels).
    public var speakerCount: Int?
    public var fraction: Double
    public var remaining: TimeInterval?
    public var phase: Phase

    public init(id: UUID, noteID: UUID, fileName: String, duration: TimeInterval? = nil,
                languageCode: String? = nil, speakerCount: Int? = nil, fraction: Double = 0,
                remaining: TimeInterval? = nil, phase: Phase = .queued) {
        self.id = id
        self.noteID = noteID
        self.fileName = fileName
        self.duration = duration
        self.languageCode = languageCode
        self.speakerCount = speakerCount
        self.fraction = fraction
        self.remaining = remaining
        self.phase = phase
    }
}

/// The file queue the sheet observes: N4's `FileTranscriptionQueue` (the one
/// consumer of each `FileJob.events`) through an adapter; tests use a double.
@MainActor
public protocol FileQueueModel: AnyObject {
    var rows: [FileQueueRow] { get }
    /// Copies each file into storage, makes its note, and queues it.
    func add(_ urls: [URL], language: SpeechLanguage, speakerLabels: Bool, summary: Bool) async
    func cancel(_ id: UUID)
}

/// "Transcribe audio files" (board 6, `Import.dc.html`): the drop zone, the
/// queue with real progress and time left, options, Close. The queue keeps
/// running after Close.
public struct TranscribeFilesSheet: View {
    /// Nil where the file queue is not wired yet: the drop zone says so.
    let queue: (any FileQueueModel)?
    let preferences: SpeechPreferences
    let onOpenNote: (UUID) -> Void
    let onClose: () -> Void
    @State private var choosesFiles = false
    @State private var isTargeted = false
    @Environment(\.mfRendersStatically) private var rendersStatically

    public init(queue: (any FileQueueModel)?, preferences: SpeechPreferences,
                onOpenNote: @escaping (UUID) -> Void, onClose: @escaping () -> Void) {
        self.queue = queue
        self.preferences = preferences
        self.onOpenNote = onOpenNote
        self.onClose = onClose
    }

    /// What AVFoundation can open with an audio track (spec 4.6).
    public static let acceptedTypes: [UTType] = [.audio, .audiovisualContent, .movie]

    /// The files among `urls` worth queueing: local files (not folders or
    /// web links) whose type is audio or video. A PDF dropped by mistake
    /// never becomes a failing transcript note.
    public static func acceptedFiles(_ urls: [URL]) -> [URL] {
        urls.filter { url in
            guard url.isFileURL, !url.hasDirectoryPath else { return false }
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                return false
            }
            guard let type = UTType(filenameExtension: url.pathExtension) else { return false }
            return acceptedTypes.contains { type.conforms(to: $0) }
        }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Transcribe audio files")
                .font(MFFont.title)
                .foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
            Text("Each file becomes a note in Audio transcripts, with the audio attached so you can play any line back.")
                .font(MFFont.body)
                .foregroundStyle(MFColor.secondary)
                .lineSpacing(MFLineSpacing.normal)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, MFSpacing.snug)
            dropZone
                .padding(.top, MFSpacing.l)
            if let queue, !queue.rows.isEmpty {
                MFSectionHeader("Queue")
                    .padding(.top, MFSpacing.roomy)
                    .padding(.bottom, MFSpacing.s)
                VStack(spacing: 0) {
                    ForEach(queue.rows) { row in
                        Rectangle().fill(MFColor.hairline).frame(height: MFLayout.hairline)
                        FileQueueRowView(row: row, onCancel: { queue.cancel(row.id) },
                                         onOpen: { onOpenNote(row.noteID) })
                    }
                }
            }
            Rectangle().fill(MFColor.hairline).frame(height: MFLayout.hairline)
                .padding(.top, MFSpacing.m)
            options
                .padding(.vertical, MFSpacing.l)
            HStack(alignment: .center) {
                Text("Keeps going if you close this. Runs on the Neural Engine, so chat stays free.")
                    .font(MFFont.callout)
                    .foregroundStyle(MFColor.secondary)
                Spacer()
                Button("Close", action: onClose)
                    .buttonStyle(MFButtonStyle(.secondary, size: .large))
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(ImportMetrics.padding)
        .frame(width: ImportMetrics.width)
        .background(MFColor.surface)
        .fileImporter(isPresented: $choosesFiles, allowedContentTypes: Self.acceptedTypes,
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { add(urls) }
        }
    }

    /// Queues the accepted files; false when none qualifies or no queue is wired.
    @discardableResult
    private func add(_ urls: [URL]) -> Bool {
        let urls = Self.acceptedFiles(urls)
        guard let queue, !urls.isEmpty else { return false }
        Task {
            await queue.add(urls, language: preferences.language, speakerLabels: preferences.fileSpeakerLabels,
                            summary: preferences.fileSummary)
        }
        return true
    }

    private var dropZone: some View {
        VStack(spacing: MFSpacing.s) {
            Image(systemName: "square.and.arrow.up")
                .font(.system(size: ImportMetrics.dropIcon, weight: .regular))
                .foregroundStyle(MFColor.secondary)
                .accessibilityHidden(true)
            HStack(spacing: MFSpacing.xs) {
                Text("Drop audio or video here, or")
                    .foregroundStyle(MFColor.ink)
                Button { choosesFiles = true } label: {
                    Text("choose files").underline().foregroundStyle(MFColor.accent)
                }
                .buttonStyle(.plain)
                .keyboardShortcut("o", modifiers: .command)
                .disabled(queue == nil)
            }
            .font(MFFont.bodyEmphasis)
            Text(queue == nil ? "Transcribing files is not available in this build yet."
                              : "m4a, mp3, wav, aiff, caf, flac, mp4, mov · Voice Memos files work too")
                .font(MFFont.preview)
                .foregroundStyle(MFColor.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, MFSpacing.xl)
        .background(RoundedRectangle(cornerRadius: MFRadius.l)
            .fill(isTargeted ? MFColor.accentFillFaint : Color.clear))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l)
            .strokeBorder(isTargeted ? MFColor.accent : MFColor.hairlineStrong,
                          style: StrokeStyle(lineWidth: ImportMetrics.dashWidth, dash: ImportMetrics.dash)))
        .modifier(FileDropTarget(isActive: !rendersStatically, isTargeted: $isTargeted) { add($0) })
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Drop audio or video files")
    }

    private var options: some View {
        @Bindable var preferences = preferences
        return HStack(spacing: MFSpacing.l) {
            Text("Language")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.secondary)
            VoiceMenuField(SpeechSettingsView.languageName(preferences.language), accessibilityLabel: "Language") {
                Button("Automatic") { preferences.language = .automatic }
                Divider()
                ForEach(VoiceFormat.languages, id: \.code) { language in
                    Button(language.name) { preferences.language = .fixed(language.code) }
                }
            }
            HStack(spacing: MFSpacing.s) {
                Toggle("Speaker labels", isOn: $preferences.fileSpeakerLabels).toggleStyle(.mf)
                Text("Speaker labels").accessibilityHidden(true)
            }
            HStack(spacing: MFSpacing.s) {
                Toggle("Write a summary", isOn: $preferences.fileSummary).toggleStyle(.mf)
                Text("Write a summary").accessibilityHidden(true)
            }
            Spacer()
        }
        .font(MFFont.callout)
        .foregroundStyle(MFColor.ink)
    }
}

/// Accepts dropped audio and video files (off in snapshots, which cannot
/// draw a drop target).
struct FileDropTarget: ViewModifier {
    let isActive: Bool
    @Binding var isTargeted: Bool
    /// Returns whether the drop was taken.
    let onDrop: ([URL]) -> Bool

    func body(content: Content) -> some View {
        if isActive {
            content.dropDestination(for: URL.self) { urls, _ in
                onDrop(urls)
            } isTargeted: { isTargeted = $0 }
        } else {
            content
        }
    }
}

/// One queue row: icon, name, the line under it, progress or state, and
/// cancel or Open note.
struct FileQueueRowView: View {
    let row: FileQueueRow
    let onCancel: () -> Void
    let onOpen: () -> Void

    var body: some View {
        HStack(spacing: MFSpacing.m) {
            icon
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(row.fileName)
                    .font(MFFont.rowTitle)
                    .foregroundStyle(MFColor.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(Self.detail(row))
                    .font(MFFont.preview)
                    .foregroundStyle(MFColor.secondary)
                    .lineLimit(1)
                if case .running = row.phase {
                    MFProgressBar(row.fraction, label: "Transcribing \(row.fileName)")
                        .frame(maxWidth: ImportMetrics.progressWidth)
                        .padding(.top, MFSpacing.xs)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
            Text(Self.state(row))
                .font(MFFont.callout)
                .foregroundStyle(Self.stateIsSuccess(row) ? MFColor.success
                                 : (Self.stateIsFailure(row) ? MFColor.danger : MFColor.secondary))
                .monospacedDigit()
            if row.phase == .done {
                Button("Open note", action: onOpen)
                    .buttonStyle(.mfSecondary)
                    .accessibilityLabel("Open the note for \(row.fileName)")
            } else if !Self.stateIsFailure(row), row.phase != .cancelled {
                Button(action: onCancel) {
                    Image(systemName: "xmark")
                        .font(MFFont.callout)
                        .frame(width: ImportMetrics.cancel, height: ImportMetrics.cancel)
                }
                .buttonStyle(.mfSecondary)
                .accessibilityLabel("Cancel \(row.fileName)")
            }
        }
        .padding(.vertical, MFSpacing.m)
    }

    @ViewBuilder
    private var icon: some View {
        Group {
            if row.phase == .done {
                Image(systemName: "checkmark")
                    .foregroundStyle(MFColor.success)
                    .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.successFill)
                        .frame(width: ImportMetrics.icon, height: ImportMetrics.icon))
            } else {
                Image(systemName: "waveform")
                    .foregroundStyle(MFColor.secondary)
                    .background(RoundedRectangle(cornerRadius: MFRadius.m).fill(MFColor.fill)
                        .frame(width: ImportMetrics.icon, height: ImportMetrics.icon))
            }
        }
        .font(MFFont.bodyEmphasis)
        .frame(width: ImportMetrics.icon, height: ImportMetrics.icon)
        .accessibilityHidden(true)
    }

    /// "1 h 18 min · Spanish, detected", "47 min · waiting", "38 min · 2 speakers · English".
    static func detail(_ row: FileQueueRow) -> String {
        var parts: [String] = []
        if let duration = row.duration { parts.append(VoiceFormat.length(duration)) }
        switch row.phase {
        case .copying: parts.append("copying")
        case .queued: parts.append("waiting")
        case .paused: parts.append("waits while dictation or a meeting runs")
        case .running:
            if let code = row.languageCode { parts.append("\(VoiceFormat.languageName(code)), detected") }
        case .done:
            if let count = row.speakerCount, count > 1 { parts.append("\(count) speakers") }
            if let code = row.languageCode { parts.append(VoiceFormat.languageName(code)) }
        case .failed(let message): parts.append(message)
        case .cancelled: parts.append("cancelled")
        }
        return parts.joined(separator: " · ")
    }

    /// "64% · about 1 min left", "Queued", "Done".
    static func state(_ row: FileQueueRow) -> String {
        switch row.phase {
        case .copying: "Copying"
        case .queued: "Queued"
        case .paused: "Paused"
        case .running:
            "\(MFProgress.percent(row.fraction) ?? 0)%" + (row.remaining.map { " · " + timeLeft($0) } ?? "")
        case .done: "Done"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }

    /// A real estimate in words: "under a minute left", "about 4 min left", "about 1 h 5 min left".
    static func timeLeft(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return "under a minute left" }
        return "about \(VoiceFormat.length(seconds)) left"
    }

    static func stateIsSuccess(_ row: FileQueueRow) -> Bool { row.phase == .done }
    static func stateIsFailure(_ row: FileQueueRow) -> Bool {
        if case .failed = row.phase { return true }
        return false
    }
}

/// Import sheet sizes from `Import.dc.html`.
enum ImportMetrics {
    static let width: CGFloat = 600
    static let padding: CGFloat = 26
    static let icon: CGFloat = 34
    /// The cancel button's glyph box (a 28 pt target with the button padding).
    static let cancel: CGFloat = 16
    static let dropIcon: CGFloat = 20
    static let dash: [CGFloat] = [5, 4]
    static let dashWidth: CGFloat = 1.5
    static let progressWidth: CGFloat = 400
}
