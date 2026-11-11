# Mference site

A single-page site for Mference: pick the memory of a Mac and see which models
that were bigger than it now run on it. No build step and no dependencies.

| File | Role |
| --- | --- |
| `index.html` | The page, with its styles and behavior |
| `data.js` | Model sizes, memory and measured decode speeds, all taken from the [README](../README.md) |
| `flux.js` | The experts grid drawn in the hero |
| `assets/icon.png` | The app icon, scaled to 256 px |

Preview it locally:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory site
```

When a number in the README changes, change the same number in `data.js`.
A speed is shown only where the README reports it for that Mac.
