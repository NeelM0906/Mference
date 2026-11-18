#!/usr/bin/env python3
"""Run the pinned Open WebUI release behind the Mference launcher.

No installed package files are changed. `check` confirms the pinned version
without loading the application or its database; `serve` runs Open WebUI's own
app on loopback. All rendering, storage, branding and tool permissions remain
Open WebUI's.
"""
import argparse
from importlib.metadata import version
import os

SUPPORTED_VERSION = "0.11.3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve", "check"])
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    installed = version("open-webui")
    if installed != SUPPORTED_VERSION:
        parser.error(f"the launcher requires Open WebUI {SUPPORTED_VERSION}; installed {installed}")
    if args.command == "check":
        print(f"Open WebUI {installed}: supported (no application/database loaded)")
        return
    if not os.environ.get("WEBUI_SECRET_KEY"):
        parser.error("WEBUI_SECRET_KEY must be supplied by the launcher")
    os.environ["FROM_INIT_PY"] = "true"
    # Import the app first: it initializes the normal Open WebUI environment
    # and middleware.
    import open_webui.main
    from open_webui.env import UVICORN_WS_PER_MESSAGE_DEFLATE
    import uvicorn

    uvicorn.run(open_webui.main.app, host=args.host, port=args.port, workers=1,
                ws_per_message_deflate=UVICORN_WS_PER_MESSAGE_DEFLATE)


if __name__ == "__main__":
    main()
