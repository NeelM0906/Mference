import AppKit
import MferenceAppCore
import MferenceDesign
import SwiftUI

/// Turning Find on (canvas board 4): the folders to search, what happens
/// next, and the one download.
struct FindSetupView: View {
    let find: FindService
    @State private var chosen: [URL: Bool] = [:]
    @State private var extra: [URL] = []
    @State private var starting = false

    private var home: URL { FileManager.default.homeDirectoryForCurrentUser }

    /// Documents, Desktop and Downloads ticked; iCloud Drive offered, off.
    private var candidates: [URL] {
        FindSettings.defaultRoots + [home.appending(path: "Library/Mobile Documents/com~apple~CloudDocs",
                                                     directoryHint: .isDirectory)] + extra
    }

    var body: some View {
        ScrollView {
            HStack(alignment: .top, spacing: MFSpacing.xxl + MFSpacing.s) {
                main.frame(maxWidth: FindMetrics.setupMaxWidth, alignment: .leading)
                steps.frame(width: FindMetrics.setupSideWidth)
            }
            .padding(.horizontal, MFSpacing.xxl + MFSpacing.xl)
            .padding(.vertical, MFSpacing.xxl + MFSpacing.s)
            .frame(maxWidth: .infinity)
        }
        .onAppear {
            for url in FindSettings.defaultRoots where chosen[url] == nil { chosen[url] = true }
        }
    }

    private var main: some View {
        VStack(alignment: .leading, spacing: MFSpacing.l) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(MFFont.largeTitle)
                .foregroundStyle(MFColor.accent)
                .frame(width: FindMetrics.setupIcon, height: FindMetrics.setupIcon)
                .background(MFColor.accentFill, in: RoundedRectangle(cornerRadius: MFRadius.l))
                .accessibilityHidden(true)
            Text("Find files by what is in them")
                .font(MFFont.largeTitle).foregroundStyle(MFColor.ink)
                .accessibilityAddTraits(.isHeader)
            Text("Describe a file the way you remember it: “the lease clause about pets”, “that screenshot with the wifi password”. Find reads your files on this Mac and keeps a private index here. Nothing is uploaded.")
                .font(MFFont.body).foregroundStyle(MFColor.secondary)
                .lineSpacing(MFLineSpacing.normal)
                .fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 0) {
                Text("FOLDERS TO SEARCH")
                    .font(MFFont.sectionHeader).tracking(MFFont.sectionHeaderTracking)
                    .foregroundStyle(MFColor.tertiary)
                    .padding(.bottom, MFSpacing.xs)
                ForEach(candidates, id: \.self) { url in
                    Toggle(isOn: Binding(get: { chosen[url] ?? false }, set: { chosen[url] = $0 })) {
                        VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                            Text(title(for: url)).font(MFFont.bodySmall).foregroundStyle(MFColor.ink)
                            if url.path.contains("Mobile Documents") {
                                Text("Downloaded files only").font(MFFont.caption).foregroundStyle(MFColor.tertiary)
                            }
                        }
                    }
                    .toggleStyle(.checkbox)
                    .padding(.vertical, MFSpacing.s)
                    Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
                }
                Button("Add a folder…", action: addFolder)
                    .buttonStyle(.mfSecondary)
                    .padding(.top, MFSpacing.compact)
            }
            Text("macOS asks once for each folder the first time Find reads it. Hidden folders, apps, caches and code projects are skipped; you can change that in Settings › Find.")
                .font(MFFont.callout).foregroundStyle(MFColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(MFSpacing.m)
                .background(MFColor.sidebar, in: RoundedRectangle(cornerRadius: MFRadius.l))
            HStack(spacing: MFSpacing.m) {
                Button("Turn on Find") {
                    starting = true
                    let roots = candidates.filter { chosen[$0] == true }
                    Task { await find.turnOn(roots: roots) }
                }
                .buttonStyle(.mfPrimary)
                .disabled(starting || !candidates.contains { chosen[$0] == true })
                Text("Downloads the search model, \(FindModel.sizeLabel)")
                    .font(MFFont.caption).foregroundStyle(MFColor.tertiary)
            }
        }
    }

    private var steps: some View {
        VStack(alignment: .leading, spacing: MFSpacing.comfortable) {
            Text("What happens next").font(MFFont.sectionTitle).foregroundStyle(MFColor.ink)
            step(1, "Words first, in minutes",
                 "Find reads the text of each file, and you can search exact words almost at once.")
            step(2, "Meaning fills in behind it",
                 "The search model learns what each passage is about. Files you opened recently go first. It works while the Mac is plugged in and you are not chatting.")
            step(3, "Stays current",
                 "New and changed files are picked up within seconds, even ones changed while Mference was closed.")
            Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
            Grid(alignment: .leading, horizontalSpacing: MFSpacing.comfortable, verticalSpacing: MFSpacing.xs) {
                GridRow {
                    Text("Model").foregroundStyle(MFColor.tertiary)
                    Text("EmbeddingGemma 2, text part · \(FindModel.licence)")
                }
                GridRow {
                    Text("Size").foregroundStyle(MFColor.tertiary)
                    Text("\(FindModel.sizeLabel) download; freed from memory when idle")
                }
            }
            .font(MFFont.callout)
            .foregroundStyle(MFColor.ink)
        }
        .padding(MFSpacing.xl)
        .background(MFColor.surface, in: RoundedRectangle(cornerRadius: MFRadius.l))
        .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.hairline))
    }

    private func step(_ number: Int, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: MFSpacing.m) {
            Text("\(number)")
                .font(MFFont.calloutSemibold)
                .foregroundStyle(MFColor.secondary)
                .frame(width: FindMetrics.stepNumber, height: FindMetrics.stepNumber)
                .background(MFColor.fill, in: Circle())
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: MFSpacing.xxs) {
                Text(title).font(MFFont.bodySmallEmphasis).foregroundStyle(MFColor.ink)
                Text(detail).font(MFFont.callout).foregroundStyle(MFColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func title(for url: URL) -> String {
        url.path.contains("Mobile Documents") ? "iCloud Drive" : url.lastPathComponent
    }

    private func addFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.prompt = "Add"
        guard panel.runModal() == .OK else { return }
        for url in panel.urls where !candidates.contains(url) {
            extra.append(url)
            chosen[url] = true
        }
    }
}
