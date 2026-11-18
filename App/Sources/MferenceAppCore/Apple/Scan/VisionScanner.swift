import CoreGraphics
import Foundation
import Vision

/// One recognized line of text.
public struct ScanLine: Sendable, Hashable, Codable {
    public var text: String
    /// Vision's confidence, 0...1.
    public var confidence: Float
    /// Normalized box, origin at the top left (for highlighting on the image).
    public var box: CGRect
    /// Baseline tilt as dy/dx in top-left coordinates (a photo held askew).
    public var slope: Double

    public init(text: String, confidence: Float, box: CGRect, slope: Double = 0) {
        self.text = text
        self.confidence = confidence
        self.box = box
        self.slope = slope
    }
}

/// What text recognition read from one image.
public struct ScanReading: Sendable, Hashable, Codable {
    /// Lines in reading order.
    public var lines: [ScanLine]
    /// Vision's transcript of the whole document.
    public var transcript: String
    /// Image width over height (normalized boxes hide it; tilt needs it).
    public var aspect: Double

    public init(lines: [ScanLine], transcript: String? = nil, aspect: Double = 1) {
        self.lines = lines
        self.transcript = transcript ?? lines.map(\.text).joined(separator: "\n")
        self.aspect = aspect
    }

    public var isEmpty: Bool { lines.isEmpty && transcript.isEmpty }

    /// The printed rows: Vision often returns a label ("TOTAL") and its
    /// amount ("44.49") as separate lines, so lines whose boxes share a
    /// baseline are joined left to right. Without boxes, the lines as given.
    public var rows: [String] {
        let boxed = lines.filter { $0.box.height > 0 }
        guard boxed.count == lines.count, !boxed.isEmpty else { return lines.map(\.text) }
        let tilt = pageTilt
        let level = { (line: ScanLine) in Double(line.box.midY) - tilt * Double(line.box.midX) }
        // A tilted line's box is taller than its text by about tilt x width.
        let height = { (line: ScanLine) in
            max(Double(line.box.height) - abs(tilt) * Double(line.box.width), Double(line.box.height) * 0.3)
        }
        var groups: [[ScanLine]] = []
        for line in lines.sorted(by: { level($0) < level($1) }) {
            if let index = groups.indices.last, let anchor = groups[index].first,
               abs(level(line) - level(anchor)) < min(height(line), height(anchor)) * 0.6 {
                groups[index].append(line)
            } else {
                groups.append([line])
            }
        }
        return groups.map { $0.sorted { $0.box.minX < $1.box.minX }.map(\.text).joined(separator: "   ") }
    }

    /// The page's tilt as dy/dx in normalized top-left coordinates, so a
    /// label and its amount far to the right still share a row. From the
    /// drift of the left text edge down the page (Theil-Sen over the
    /// left-aligned lines), else the median baseline slope.
    var pageTilt: Double {
        let leftmost = lines.map { Double($0.box.minX) }.min() ?? 0
        let edge = lines.filter { Double($0.box.minX) < leftmost + 0.08 }
        if edge.count >= 4 {
            var drifts: [Double] = []
            for i in edge.indices {
                for j in edge.indices where j > i {
                    let dy = Double(edge[j].box.midY - edge[i].box.midY)
                    if abs(dy) > 0.02 { drifts.append(Double(edge[j].box.minX - edge[i].box.minX) / dy) }
                }
            }
            if !drifts.isEmpty {
                let drift = drifts.sorted()[drifts.count / 2]
                // Rotating by t moves the left edge by tan t per unit of
                // height and rows by -tan t per unit of width, in pixels.
                return max(-0.3, min(0.3, -drift * aspect * aspect))
            }
        }
        let slopes = lines.filter { $0.box.width > 0.15 }.map(\.slope).sorted()
        return slopes.isEmpty ? 0 : slopes[slopes.count / 2]
    }
}

public enum ScanError: Error, Sendable, Hashable {
    case unreadableImage
    case noTextFound
    case recognitionFailed(String)

    public var userMessage: String {
        switch self {
        case .unreadableImage: "This file is not an image Mference can read."
        case .noTextFound: "No text was found in this image."
        case .recognitionFailed(let detail): "Text recognition failed: \(detail)"
        }
    }
}

/// Reads text from an image on this Mac.
public protocol DocumentRecognizing: Sendable {
    func recognize(imageAt url: URL) async throws -> ScanReading
    func recognize(image: CGImage) async throws -> ScanReading
}

/// Apple Vision's `RecognizeDocumentsRequest` (decision L12). Runs on this
/// Mac with no permission and no network.
public struct VisionScanner: DocumentRecognizing {
    public init() {}

    public func recognize(imageAt url: URL) async throws -> ScanReading {
        guard let image = Self.loadImage(at: url) else { throw ScanError.unreadableImage }
        return try await recognize(image: image)
    }

    /// The image at `url`, or the first page of a PDF rendered at 200 dpi.
    public static func loadImage(at url: URL) -> CGImage? {
        if url.pathExtension.lowercased() == "pdf" {
            return renderFirstPage(of: url)
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }

    static func renderFirstPage(of url: URL) -> CGImage? {
        guard let document = CGPDFDocument(url as CFURL), let page = document.page(at: 1) else { return nil }
        let bounds = page.getBoxRect(.mediaBox)
        let scale = 200.0 / 72.0
        let width = Int(bounds.width * scale), height = Int(bounds.height * scale)
        guard width > 0, height > 0, width * height <= 60_000_000,
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: -bounds.minX, y: -bounds.minY)
        context.drawPDFPage(page)
        return context.makeImage()
    }

    public func recognize(image: CGImage) async throws -> ScanReading {
        var request = RecognizeDocumentsRequest()
        request.textRecognitionOptions.useLanguageCorrection = true
        request.textRecognitionOptions.automaticallyDetectLanguage = true
        let observations: [DocumentObservation]
        do {
            observations = try await request.perform(on: image)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw ScanError.recognitionFailed(error.localizedDescription)
        }
        var lines: [ScanLine] = []
        var transcripts: [String] = []
        for observation in observations {
            let text = observation.document.text
            transcripts.append(text.transcript)
            for line in text.lines {
                let candidate = line.topCandidates(1).first
                let string = (candidate?.string ?? line.transcript).trimmingCharacters(in: .whitespaces)
                guard !string.isEmpty else { continue }
                let box = line.boundingBox.cgRect
                let run = line.bottomRight.x - line.bottomLeft.x
                let slope = run > 0.01 ? -Double(line.bottomRight.y - line.bottomLeft.y) / Double(run) : 0
                lines.append(ScanLine(text: string, confidence: candidate?.confidence ?? line.confidence,
                                      box: CGRect(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height),
                                      slope: slope))
            }
        }
        let reading = ScanReading(lines: lines, transcript: transcripts.joined(separator: "\n"),
                                  aspect: Double(image.width) / Double(max(image.height, 1)))
        guard !reading.isEmpty else { throw ScanError.noTextFound }
        return reading
    }
}
