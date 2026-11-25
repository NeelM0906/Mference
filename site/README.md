# Mference site

The product site for Mference: a personal AI that runs on your Mac. Five static
pages with no build step and no third-party requests (the Geist font is served
from `assets/fonts`).

| File | Role |
| --- | --- |
| `index.html` | Home: hero, product tour, features, privacy, models, FAQ |
| `features.html` | Chat, Find, Mail, Calendar, Notes and Scan in detail |
| `privacy.html` | What stays on the Mac and every way the app uses the network |
| `models.html` | Supported models, the memory picker, and how to get a model |
| `developers.html` | Build from source, install models, engine, server and CLI |
| `assets/css/site.css` | All styles. Colors follow the app's `MFColor` tokens, light and dark |
| `assets/js/site.js` | The tour, Find demo, Offline switch, memory picker and copy buttons |
| `data.js` | Model sizes, memory and measured decode speeds, all taken from the [README](../README.md) |
| `assets/shots/` | The app's design renders, copied from `docs/assets/readme/` |
| `assets/fonts/` | Geist (OFL, see `OFL.txt`), copied from `App/Resources/Fonts/` |

Preview it locally:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory site
```

When a number in the README changes, change the same number in `data.js`.
A speed is shown only where the README reports it for that Mac. When a render
in `docs/assets/readme/` changes, copy it into `assets/shots/`.

## Publishing

`.github/workflows/pages.yml` publishes `site/` to GitHub Pages on every change to
`site/` on `main`. A repository admin sets the Pages source to "GitHub Actions"
once in Settings > Pages.
