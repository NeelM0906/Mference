import MferenceAppCore
import MferenceDesign
import SwiftUI

/// The Find screen (canvas board 2): filters on the left, the search box
/// and results in the middle, the preview on the right. Before Find is on
/// it shows the setup (board 4).
public struct FindView: View {
    @Environment(FindService.self) private var find: FindService?
    let openChat: (URL) -> Void
    let askAcross: (String) -> Void

    /// `openChat` starts a chat about one file; `askAcross` starts one with
    /// Files on and the search as the question.
    public init(openChat: @escaping (URL) -> Void = { _ in }, askAcross: @escaping (String) -> Void = { _ in }) {
        self.openChat = openChat
        self.askAcross = askAcross
    }

    public var body: some View {
        if let find {
            if find.settings.enabled {
                FindSearchScreen(find: find, openChat: openChat, askAcross: askAcross)
            } else {
                FindSetupView(find: find)
            }
        } else {
            SectionPlaceholderView(section: .find)
        }
    }
}

struct FindSearchScreen: View {
    let find: FindService
    let openChat: (URL) -> Void
    let askAcross: (String) -> Void
    @State private var model: FindViewModel
    @FocusState private var focused: Bool

    init(find: FindService, openChat: @escaping (URL) -> Void, askAcross: @escaping (String) -> Void) {
        self.find = find
        self.openChat = openChat
        self.askAcross = askAcross
        _model = State(initialValue: FindViewModel(find: find))
    }

    var body: some View {
        HStack(spacing: 0) {
            filters
            Rectangle().fill(MFColor.hairline).frame(width: MFLayout.hairline)
            VStack(spacing: 0) {
                header
                Rectangle().fill(MFColor.hairline).frame(height: MFLayout.hairline)
                HStack(spacing: 0) {
                    results
                    if let selected = model.selected {
                        Rectangle().fill(MFColor.hairline).frame(width: MFLayout.hairline)
                        FindPreview(result: selected, askAbout: openChat)
                            .padding(MFSpacing.l)
                            .frame(width: FindMetrics.previewWidth)
                            .frame(maxHeight: .infinity, alignment: .top)
                            .background(MFColor.canvas)
                    }
                }
            }
        }
        .onAppear { focused = true }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: MFSpacing.compact) {
            HStack(spacing: MFSpacing.compact) {
                Image(systemName: "magnifyingglass").foregroundStyle(MFColor.secondary).accessibilityHidden(true)
                TextField("Describe a file: “the lease clause about pets”", text: $model.query)
                    .textFieldStyle(.plain)
                    .font(MFFont.body)
                    .focused($focused)
                    .onSubmit { model.searchNow() }
                    .onKeyPress(.downArrow) { model.moveSelection(by: 1); return .handled }
                    .onKeyPress(.upArrow) { model.moveSelection(by: -1); return .handled }
                    .accessibilityLabel("Search files")
                if model.isSearching { ProgressView().controlSize(.small) }
                Text(model.summary).font(MFFont.caption).foregroundStyle(MFColor.tertiary)
            }
            .padding(.horizontal, MFSpacing.comfortable)
            .frame(height: FindMetrics.searchHeight)
            .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.m))
            .overlay(RoundedRectangle(cornerRadius: MFRadius.m).strokeBorder(MFColor.hairlineStrong))
            FindStatusBanner(status: find.status, modelState: find.modelState, mediaModelState: find.mediaModelState)
            if !model.results.strong.isEmpty {
                askBar
            }
        }
        .padding(.horizontal, MFSpacing.l)
        .padding(.vertical, MFSpacing.m)
    }

    private var askBar: some View {
        HStack(spacing: MFSpacing.compact) {
            Image(systemName: "bubble.left").foregroundStyle(MFColor.accent).accessibilityHidden(true)
            Text("**Ask across these \(model.results.strong.count) files.** Chat answers from them, with sources.")
                .font(MFFont.callout)
                .foregroundStyle(MFColor.ink)
            Spacer(minLength: MFSpacing.s)
            Button("Ask in Chat") { askAcross(model.query) }
            .buttonStyle(.mfPrimary)
        }
        .padding(.horizontal, MFSpacing.m)
        .padding(.vertical, MFSpacing.s)
        .background(MFColor.accentFillFaint, in: RoundedRectangle(cornerRadius: MFRadius.l))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.accentFill))
    }

    @ViewBuilder
    private var results: some View {
        if model.query.trimmingCharacters(in: .whitespaces).isEmpty {
            FindEmptyHint(status: find.status)
        } else if model.results.strong.isEmpty && !model.isSearching {
            FindNothingFound(model: model)
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: MFSpacing.xxs) {
                    ForEach(model.visible) { result in
                        FindResultRow(result: result, isSelected: result.url == model.selected?.url)
                            .onTapGesture { model.selection = result.url }
                            .onTapGesture(count: 2) { FindFiles.open(result.url) }
                    }
                    if !model.results.weak.isEmpty, !model.showsWeak {
                        Button {
                            model.showsWeak = true
                        } label: {
                            Label("\(model.results.weak.count) weaker matches", systemImage: "chevron.right")
                                .font(MFFont.callout)
                                .foregroundStyle(MFColor.secondary)
                        }
                        .buttonStyle(.plain)
                        .padding(MFSpacing.m)
                    }
                }
                .padding(MFSpacing.s)
            }
            .frame(maxWidth: .infinity)
        }
    }

    private var filters: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MFSpacing.l) {
                filterGroup("Kind") {
                    ForEach(FindKindFilter.allCases) { filter in
                        filterRow(filter.title, count: model.results.query.isEmpty ? nil : model.count(filter),
                                  selected: model.kind == filter) { model.kind = filter }
                    }
                }
                filterGroup("Modified") {
                    ForEach(FindDateFilter.allCases) { filter in
                        filterRow(filter.title, selected: model.modified == filter) { model.modified = filter }
                    }
                }
                filterGroup("Folders") {
                    filterRow("All indexed", count: find.settings.roots.count, selected: model.folder == nil) {
                        model.folder = nil
                    }
                    ForEach(find.settings.roots, id: \.self) { root in
                        filterRow(root.lastPathComponent, selected: model.folder == root) { model.folder = root }
                    }
                }
            }
            .padding(.horizontal, MFSpacing.compact)
            .padding(.vertical, MFSpacing.l)
        }
        .frame(width: FindMetrics.filterWidth)
        .background(MFColor.sidebar)
    }

    private func filterGroup<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
            Text(title.uppercased())
                .font(MFFont.sectionHeader)
                .tracking(MFFont.sectionHeaderTracking)
                .foregroundStyle(MFColor.tertiary)
                .padding(.horizontal, MFSpacing.compact)
                .padding(.bottom, MFSpacing.xs)
                .accessibilityAddTraits(.isHeader)
            content()
        }
    }

    private func filterRow(_ title: String, count: Int? = nil, selected: Bool,
                           action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Text(title).font(selected ? MFFont.bodySmallEmphasis : MFFont.bodySmall)
                Spacer()
                if let count { Text("\(count)").font(MFFont.caption).foregroundStyle(MFColor.tertiary) }
            }
            .foregroundStyle(MFColor.ink)
            .padding(.horizontal, MFSpacing.compact)
            .padding(.vertical, MFSpacing.snug)
            .background(selected ? MFColor.sidebarSelected : .clear, in: RoundedRectangle(cornerRadius: MFRadius.s))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// Before the first query: what Find can do, and its progress.
struct FindEmptyHint: View {
    let status: FindIndexStatus

    var body: some View {
        VStack(spacing: MFSpacing.m) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(MFFont.largeTitle)
                .foregroundStyle(MFColor.tertiary)
                .accessibilityHidden(true)
            Text("Describe the file you are looking for")
                .font(MFFont.headline).foregroundStyle(MFColor.ink)
            Text("“the lease clause about pets” · “that screenshot with the wifi password” · “Q3 revenue for the board”")
                .font(MFFont.callout).foregroundStyle(MFColor.secondary).multilineTextAlignment(.center)
            Text("\(status.totalFiles.formatted()) files on this Mac · nothing leaves it · ⌥Space opens Find from any app")
                .font(MFFont.caption).foregroundStyle(MFColor.tertiary)
        }
        .padding(MFSpacing.xxl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// The honest "nothing found" state (board 6B).
struct FindNothingFound: View {
    let model: FindViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: MFSpacing.m) {
            Text("No file is a close match.").font(MFFont.headline).foregroundStyle(MFColor.ink)
            Text("Find shows only results it is confident about, so you are not handed ten guesses.")
                .font(MFFont.callout).foregroundStyle(MFColor.secondary)
            HStack(spacing: MFSpacing.s) {
                if model.kind != .all {
                    Button("Search all kinds") { model.kind = .all }.buttonStyle(.mfSecondary)
                }
                if model.modified != .anyTime {
                    Button("Any date") { model.modified = .anyTime }.buttonStyle(.mfSecondary)
                }
                if !model.results.weak.isEmpty {
                    Button("Show \(model.results.weak.count) weak guesses") { model.showsWeak = true }
                        .buttonStyle(.mfSecondary)
                }
            }
            if model.find.status.phase == .learning || model.find.status.phase == .reading {
                Text("Find is still reading your files, so a match may appear soon.")
                    .font(MFFont.caption).foregroundStyle(MFColor.tertiary)
            }
        }
        .padding(MFSpacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
