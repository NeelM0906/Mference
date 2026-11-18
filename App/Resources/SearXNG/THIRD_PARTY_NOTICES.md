# Third-party software in the bundled SearXNG runtime

Shipped inside the app at `Contents/Resources/SearXNG/runtime/` (paths below are
relative to it; see `SOURCES.md` for
the exact downloads and checksums).

| Component | Licence | Where the licence text is |
| --- | --- | --- |
| SearXNG (unmodified, commit in `SOURCES.md`) | AGPL-3.0-or-later | `licenses/SearXNG-AGPL-3.0.txt`; complete source in `source/` |
| CPython (python-build-standalone build) | PSF License, with the third-party notices CPython itself carries | `licenses/Python-PSF.txt` |
| The 31 Python packages in `wheels.lock` | each package's own licence (MIT, BSD, Apache-2.0, PSF, MPL-2.0 for `certifi`, ...) | `licenses/wheels/<package>/`, and each package's `*.dist-info` |

## Native libraries in the Python runtime

python-build-standalone links these libraries statically into the
interpreter's extension modules (from the release's `PYTHON.json`). Their
licence texts come from the project's full archive of the same release
(`SOURCES.md`) and ship at
`Contents/Resources/SearXNG/licenses/python-build-standalone/` (in the
repository: `App/Resources/SearXNG/licenses/python-build-standalone/`).
Nothing else from that archive ships.

| Library | Used by | Licence text |
| --- | --- | --- |
| OpenSSL 3 | `_ssl`, `_hashlib` | `LICENSE.openssl-3.txt` (Apache-2.0) |
| SQLite | `_sqlite3` | `LICENSE.sqlite.txt` (public domain) |
| XZ / liblzma | `_lzma` | `LICENSE.liblzma.txt` |
| bzip2 | `_bz2` | `LICENSE.bzip2.txt` |
| libffi | `_ctypes` | `LICENSE.libffi.txt` |
| mpdecimal | `_decimal` | `LICENSE.mpdecimal.txt` |
| Expat | `pyexpat` | `LICENSE.expat.txt` |
| libuuid | `_uuid` | `LICENSE.libuuid.txt` |

`readline`, `_curses` and `zlib` use macOS's own libedit, ncurses and zlib,
which are not bundled. Tcl/Tk (statically stubbed for `_tkinter`) is removed
by `prepare.sh` because SearXNG does not use it.
