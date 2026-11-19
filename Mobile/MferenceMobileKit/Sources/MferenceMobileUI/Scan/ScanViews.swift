import MferenceDesign
import SwiftUI

/// Scan on iPhone and Duo closed: the camera on top, the fields in a panel
/// below.
struct ScanScreen: View {
    @Bindable var model: MobileAppModel

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                ScanCameraView(model: model)
                    .frame(height: model.scanPhase == .viewfinder ? geometry.size.height : geometry.size.height * 0.46)
                if model.scanPhase != .viewfinder {
                    ScanFieldsView(model: model, compact: true)
                        .padding(.horizontal, MFSpacing.xl - 4)
                        .padding(.top, MFSpacing.s)
                        .frame(maxHeight: .infinity, alignment: .top)
                        .background(UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28)
                            .fill(MFColor.surface)
                            .ignoresSafeArea(edges: .bottom))
                        .offset(y: -24)
                        .padding(.bottom, -24)
                        .transition(.move(edge: .bottom))
                }
            }
            .animation(.snappy(duration: 0.25), value: model.scanPhase)
        }
        .background(ScanCameraView.darkGround.ignoresSafeArea())
    }
}

/// The camera area. The simulator has no camera, so it shows the sample
/// receipt; the shutter "reads" it again.
struct ScanCameraView: View {
    @Bindable var model: MobileAppModel

    static let darkGround = Color(red: 0.086, green: 0.09, blue: 0.106)

    var body: some View {
        ZStack {
            RadialGradient(colors: [Color(red: 0.165, green: 0.172, blue: 0.2), Self.darkGround],
                           center: .center, startRadius: 10, endRadius: 360)
            VStack(spacing: MFSpacing.l) {
                HStack {
                    if model.scanPhase != .viewfinder {
                        Button("Retake") { model.retake() }
                            .font(MFPhoneFont.previewEmphasis)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .frame(height: 40)
                            .background(Capsule().fill(.white.opacity(0.14)))
                    }
                    Spacer()
                    Label(model.scanPhase == .reading ? "Reading…" : "Read on this iPhone", systemImage: "lock")
                        .font(MFPhoneFont.control)
                        .foregroundStyle(Color(red: 0.435, green: 0.827, blue: 0.627))
                        .padding(.horizontal, MFSpacing.m)
                        .frame(height: 32)
                        .background(Capsule().fill(Color(red: 0.09, green: 0.208, blue: 0.153).opacity(0.9)))
                }
                Spacer(minLength: 0)
                SampleReceipt(isReading: model.scanPhase == .reading)
                    .scaleEffect(model.scanPhase == .viewfinder ? 1.25 : 1)
                Spacer(minLength: 0)
                SegmentedPills(options: ScanKind.allCases, selection: $model.scanKind, title: \.rawValue, onDark: true)
                if model.scanPhase == .viewfinder {
                    Button {
                        model.capture()
                    } label: {
                        Circle().fill(.white).padding(5)
                            .overlay(Circle().strokeBorder(.white, lineWidth: 4))
                            .frame(width: 72, height: 72)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Take photo")
                    .padding(.bottom, MFSpacing.xl)
                }
            }
            .padding(.horizontal, MFPhoneMetrics.screenMargin)
            .padding(.top, MFSpacing.s)
            .padding(.bottom, MFSpacing.xxl)
        }
        .clipped()
    }
}

/// The sample receipt with the detected outline.
struct SampleReceipt: View {
    var isReading = false

    var body: some View {
        VStack(spacing: 4) {
            Text("BLUE DOOR CAFE").font(.system(size: 12, weight: .semibold, design: .monospaced))
            Text("12:48").font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            Group {
                line("Lunch special x4", "68.00").padding(.top, 6)
                line("Iced tea x4", "14.00")
                Rectangle().stroke(style: StrokeStyle(lineWidth: 1, dash: [3])).frame(height: 1).opacity(0.3)
                line("Tax", "7.07")
                line("Tip", "16.40")
                line("TOTAL", "105.47").fontWeight(.semibold)
            }
            .font(.system(size: 10, design: .monospaced))
        }
        .foregroundStyle(Color(red: 0.09, green: 0.094, blue: 0.11))
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(width: 190)
        .background(Color(red: 1, green: 0.992, blue: 0.965))
        .overlay(Rectangle().strokeBorder(Color(red: 0.561, green: 0.635, blue: 1), lineWidth: 3))
        .overlay {
            if isReading {
                ProgressView().tint(Color(red: 0.184, green: 0.278, blue: 0.722))
            }
        }
        .rotationEffect(.degrees(-2))
        .shadow(color: .black.opacity(0.4), radius: 15, y: 10)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Receipt from Blue Door Cafe")
    }

    private func line(_ item: String, _ amount: String) -> some View {
        HStack {
            Text(item)
            Spacer()
            Text(amount)
        }
    }
}

/// The fields read from the scan, the destination and Add row.
struct ScanFieldsView: View {
    @Bindable var model: MobileAppModel
    var compact: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if compact {
                Capsule().fill(MFColor.hairlineStrong).frame(width: 36, height: 5)
                    .frame(maxWidth: .infinity)
                    .padding(.bottom, MFSpacing.m)
            }
            HStack(spacing: MFSpacing.s) {
                Text(model.scanMerchant)
                    .font(MFPhoneFont.title)
                    .foregroundStyle(MFColor.ink)
                Spacer()
                if model.scanPhase == .reading {
                    ProgressView()
                } else if !compact {
                    PhoneChip(title: "Read on this iPhone", tone: .success)
                }
            }
            .padding(.bottom, MFSpacing.s)
            if !compact {
                Text("Built-in text recognition read the receipt; the on-device model filled the fields. Check them, then choose where the row goes.")
                    .font(MFPhoneFont.preview)
                    .foregroundStyle(MFColor.secondary)
                    .padding(.bottom, MFSpacing.s)
            }
            ForEach($model.scanFields) { $field in
                HStack(spacing: MFSpacing.compact) {
                    Text(field.label)
                        .font(MFPhoneFont.callout)
                        .foregroundStyle(MFColor.secondary)
                        .frame(width: 84, alignment: .leading)
                    TextField(field.label, text: $field.value)
                        .font(field.id == "total" ? MFPhoneFont.rowTitle : MFPhoneFont.callout)
                        .foregroundStyle(MFColor.ink)
                        .padding(.horizontal, field.isSure ? 0 : MFSpacing.compact)
                        .frame(height: 36)
                        .background(RoundedRectangle(cornerRadius: MFRadius.m)
                            .fill(field.isSure ? Color.clear : MFColor.approvalFill))
                        .overlay(RoundedRectangle(cornerRadius: MFRadius.m)
                            .strokeBorder(field.isSure ? Color.clear : MFColor.approvalBorder, lineWidth: MFLayout.hairline))
                    PhoneChip(title: field.isSure ? "Sure" : "Check", tone: field.isSure ? .success : .preview)
                }
                .frame(minHeight: MFPhoneMetrics.touchTarget)
                .overlay(alignment: .bottom) {
                    Rectangle().fill(MFColor.hairlineFaint).frame(height: MFLayout.hairline)
                }
            }
            if model.scanPhase == .added {
                Label("Added to Expenses 2026, row 42", systemImage: "checkmark.circle.fill")
                    .font(MFPhoneFont.button)
                    .foregroundStyle(MFColor.success)
                    .frame(maxWidth: .infinity, minHeight: MFPhoneMetrics.primaryButtonHeight)
                    .background(Capsule().fill(MFColor.successFill))
                    .padding(.top, MFSpacing.l)
            } else {
                HStack(spacing: MFSpacing.compact) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Add as a row to Expenses 2026").font(MFFont.geist(15, .semibold, relativeTo: .subheadline))
                            .foregroundStyle(MFColor.ink)
                        Text("CSV in Files · after row 41").font(MFPhoneFont.caption).foregroundStyle(MFColor.secondary)
                    }
                    Spacer(minLength: 0)
                    Button("Change") {}
                        .buttonStyle(PhoneButtonStyle(kind: .secondary, size: .compact))
                    if !compact {
                        Button("Add row") { model.addScanRow() }
                            .buttonStyle(PhoneButtonStyle(kind: .primary, size: .compact))
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, MFSpacing.compact)
                .background(RoundedRectangle(cornerRadius: MFRadius.l).fill(MFColor.accentFillFaint))
                .overlay(RoundedRectangle(cornerRadius: MFRadius.l).strokeBorder(MFColor.accentFill, lineWidth: MFLayout.hairline))
                .padding(.top, MFSpacing.m)
                if compact {
                    Button("Add row") { model.addScanRow() }
                        .buttonStyle(PhoneButtonStyle(kind: .primary, size: .large, fullWidth: true))
                        .padding(.top, MFSpacing.m)
                }
            }
        }
        .disabled(model.scanPhase == .reading)
        .sensoryFeedback(.success, trigger: model.scanPhase == .read)
    }
}
