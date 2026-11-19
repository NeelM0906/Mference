# Geist (bundled fonts)

Geist 400/500/600/700, static OpenType, bundled under the SIL Open Font
License 1.1 (`OFL.txt`, copied unchanged from the release). Approved download:
App Program spec decision P6.

- Release: Geist v1.9.2, the official Vercel release
  (https://github.com/vercel/geist-font/releases/tag/v1.9.2), published 2026-11-06.
- Archive: https://github.com/vercel/geist-font/releases/download/v1.9.2/geist-font-v1.9.2.zip
  SHA-256 `15f066d58dcb99fb10014a49913009fdaf41e464cfb8851a44b44e593c95d3d8`
- Files, copied unchanged from `geist-font/Geist/otf/` and `geist-font/OFL.txt`
  in that archive:

| File | Weight | SHA-256 |
| --- | --- | --- |
| `Geist-Regular.otf` | 400 | `674ccb412bf5f6bc75bb1658189f83daf6a4046d742da64bdd46e671fbd779e0` |
| `Geist-Medium.otf` | 500 | `bd73ca3934c6502c6f865bd9b630ab6f6f59c16478824aefaee3e2da664d6a58` |
| `Geist-SemiBold.otf` | 600 | `e7516c6998318861f48335d94ba1fc485cdd13eae46f1144e58cddef80e0e93e` |
| `Geist-Bold.otf` | 700 | `28fb5431ce38bfe1ff76f4ff19494f79bfcfad6c31a454f736e2911f04fc1f53` |
| `OFL.txt` | licence | `021caa749f89fecd7c2e40a50499de29e84d2bc19251659b47d499c2fbfffc17` |

`Scripts/bundle.sh` copies this directory to `Mference.app/Contents/Resources/Fonts`;
`MFFontRegistry.registerBundledFonts()` registers the four faces for the
process at launch so `MFFont` resolves to Geist. `DesignFontTests` checks each
file's hash against its row above.
