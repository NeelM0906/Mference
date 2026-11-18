# Third-party software in the bundled SearXNG runtime

Shipped inside the app at `Contents/Resources/SearXNG/runtime/` (paths below are
relative to it; see `SOURCES.md` for
the exact downloads and checksums).

| Component | Licence | Where the licence text is |
| --- | --- | --- |
| SearXNG (unmodified, commit in `SOURCES.md`) | AGPL-3.0-or-later | `licenses/SearXNG-AGPL-3.0.txt`; complete source in `source/` |
| CPython (python-build-standalone build) | PSF License, with the third-party notices CPython itself carries | `licenses/Python-PSF.txt` |
| The 31 Python packages in `wheels.lock` | each package's own licence (MIT, BSD, Apache-2.0, PSF, MPL-2.0 for `certifi`, ...) | `licenses/wheels/<package>/`, and each package's `*.dist-info` |

python-build-standalone links further native libraries statically into the
interpreter (among them OpenSSL, SQLite, XZ, bzip2, zlib, libffi and the
terminal libraries used by `readline`). Their licence texts are not part of
the `install_only_stripped` archive this app pins; they ship in the
project's full archives and are listed in its documentation
(https://gregoryszorc.com/docs/python-build-standalone/main/). Adding those
texts here is an open item: it needs one more download (the matching full
archive) and is tracked in the PR that introduced this file. Tcl/Tk, which
the runtime also contains, is removed by `prepare.sh` because SearXNG does
not use it.
