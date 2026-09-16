#!/usr/bin/env python3
"""Run pinned Open WebUI with Swift-Qwen reasoning-history compatibility.

No installed package files are changed. The default Open WebUI 0.11.3 policy
drops reasoning when replaying generic OpenAI-compatible messages, including
internal native tool loops. Override only that policy, only for our Swift ID.
All rendering, storage, branding and tool permissions remain Open WebUI's.
"""
import argparse
from importlib.metadata import version
import os

SWIFT_ID = "swift-qwen3.8-27b-int4g64"
SUPPORTED_VERSION = "0.11.3"


def reasoning_format(original, model):
    # Open WebUI wraps the server's identity under `openai` when merging
    # connections, replacing its top-level owner with "openai".
    source = model.get("openai") or model
    model_id = source.get("id", "").split("@", 1)[0]
    if source.get("owned_by") == "mference" and model_id == SWIFT_ID:
        return "reasoning_content"
    return original(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", choices=["serve"])
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    installed = version("open-webui")
    if installed != SUPPORTED_VERSION:
        parser.error(f"reasoning-history adapter requires Open WebUI {SUPPORTED_VERSION}; installed {installed}")
    if not os.environ.get("WEBUI_SECRET_KEY"):
        parser.error("WEBUI_SECRET_KEY must be supplied by the launcher")
    os.environ["FROM_INIT_PY"] = "true"
    # Import the app first: it initializes the normal Open WebUI environment
    # and middleware. One worker preserves this process-local override.
    import open_webui.main
    from open_webui.utils import middleware
    from open_webui.env import UVICORN_WS_PER_MESSAGE_DEFLATE
    import uvicorn

    original = middleware.get_reasoning_format
    middleware.get_reasoning_format = lambda model: reasoning_format(original, model)
    uvicorn.run(open_webui.main.app, host=args.host, port=args.port, workers=1,
                ws_per_message_deflate=UVICORN_WS_PER_MESSAGE_DEFLATE)


if __name__ == "__main__":
    main()
