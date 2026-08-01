// Renders the Mference app icon: the Bounded Flux mark reduced to what
// survives at 16 points. Sixteen ring segments stand for the sixteen routed
// expert slots each layer holds; the M is the form the admitted streams
// condense into. Deterministic — no randomness, so the icon is reproducible.
//
//   swift design/RenderIcon.swift <output.png> [size]

import AppKit
import CoreGraphics
import Foundation

let outputPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1] : "icon.png"
let size = CommandLine.arguments.count > 2
    ? (Int(CommandLine.arguments[2]) ?? 1024) : 1024

let S = CGFloat(size)
let colorSpace = CGColorSpaceCreateDeviceRGB()
guard let ctx = CGContext(data: nil, width: size, height: size,
                          bitsPerComponent: 8, bytesPerRow: 0,
                          space: colorSpace,
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
else { fatalError("could not create bitmap context") }

func rgb(_ r: Int, _ g: Int, _ b: Int, _ a: CGFloat = 1) -> CGColor {
    CGColor(red: CGFloat(r) / 255, green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255, alpha: a)
}

let ink = rgb(15, 15, 18)
let ember = rgb(217, 119, 87)
let core = rgb(240, 198, 116)
let faint = rgb(70, 80, 94)

// Rounded-rect plate, macOS icon proportions.
let inset = S * 0.055
let plate = CGRect(x: inset, y: inset, width: S - 2 * inset, height: S - 2 * inset)
let platePath = CGPath(roundedRect: plate,
                       cornerWidth: S * 0.2237, cornerHeight: S * 0.2237,
                       transform: nil)
ctx.addPath(platePath)
ctx.setFillColor(ink)
ctx.fillPath()

// Everything else is clipped to the plate.
ctx.addPath(platePath)
ctx.clip()

let cx = S / 2, cy = S / 2

// A faint radial wash: the reservoir of streams that never light.
if let wash = CGGradient(colorsSpace: colorSpace,
                         colors: [rgb(70, 80, 94, 0.30), rgb(15, 15, 18, 0)] as CFArray,
                         locations: [0, 1]) {
    ctx.drawRadialGradient(wash,
                           startCenter: CGPoint(x: cx, y: cy), startRadius: S * 0.10,
                           endCenter: CGPoint(x: cx, y: cy), endRadius: S * 0.46,
                           options: [])
}

// The bounded window: sixteen segments, one per expert slot. Luminance varies
// so the ring reads as a memory of uneven traffic rather than a plain circle.
let slots = 16
let ringR = S * 0.352
let gap: CGFloat = 0.052
let heat: [CGFloat] = [0.30, 0.95, 0.55, 0.20, 0.70, 0.35, 1.00, 0.45,
                       0.25, 0.80, 0.40, 0.60, 0.30, 0.90, 0.50, 0.22]
ctx.setLineCap(.butt)
for i in 0..<slots {
    let a0 = (CGFloat.pi * 2 * CGFloat(i)) / CGFloat(slots) + gap
    let a1 = (CGFloat.pi * 2 * CGFloat(i + 1)) / CGFloat(slots) - gap
    let h = heat[i]
    let c = CGColor(red: (70 + (217 - 70) * h) / 255,
                    green: (80 + (119 - 80) * h) / 255,
                    blue: (94 + (87 - 94) * h) / 255,
                    alpha: 0.34 + 0.62 * h)
    ctx.setStrokeColor(c)
    ctx.setLineWidth(S * 0.0225)
    ctx.addArc(center: CGPoint(x: cx, y: cy), radius: ringR,
               startAngle: a0, endAngle: a1, clockwise: false)
    ctx.strokePath()
}

// The M. Same skeleton the generative sketch attracts toward, drawn directly
// here because an icon has to be legible at 16 points, not emergent.
let hw = S * 0.163          // half width
let top = cy + S * 0.121    // CG origin is bottom-left
let bot = cy - S * 0.131
let mid = cy - S * 0.054
let m = CGMutablePath()
m.move(to: CGPoint(x: cx - hw, y: bot))
m.addLine(to: CGPoint(x: cx - hw, y: top))
m.addLine(to: CGPoint(x: cx, y: mid))
m.addLine(to: CGPoint(x: cx + hw, y: top))
m.addLine(to: CGPoint(x: cx + hw, y: bot))

ctx.setLineJoin(.round)
ctx.setLineCap(.round)

// Ember bloom underneath, then the bright core stroke on top.
ctx.setShadow(offset: .zero, blur: S * 0.055, color: ember)
ctx.addPath(m)
ctx.setStrokeColor(ember)
ctx.setLineWidth(S * 0.088)
ctx.strokePath()
ctx.setShadow(offset: .zero, blur: 0, color: nil)

ctx.addPath(m)
ctx.setStrokeColor(core)
ctx.setLineWidth(S * 0.050)
ctx.strokePath()

guard let image = ctx.makeImage() else { fatalError("could not render image") }
let rep = NSBitmapImageRep(cgImage: image)
guard let png = rep.representation(using: .png, properties: [:]) else {
    fatalError("could not encode PNG")
}
try png.write(to: URL(fileURLWithPath: outputPath))
print("wrote \(outputPath) at \(size)x\(size)")
