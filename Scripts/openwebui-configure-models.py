#!/usr/bin/env python3
"""Register every Mference model in Open WebUI with builtin tools switched off.

Open WebUI 0.11 defaults every model to "native function calling" and attaches
its builtin tool schemas (web search, code interpreter, memory, notes, time,
ask-user, and more) to each chat request. MferenceServer renders those tools
into the prompt as any OpenAI-compatible server must, which turned a 27-token
question into a 5,445-token prompt and a 2-second answer into a 45-second one.
There is no environment variable for this in 0.11.3; it is a per-model
capability. This script sets it, idempotently, for every model the Mference
server advertises, so the launcher can run it on every start and newly
installed models pick it up automatically. Re-enable a model's builtin tools in
Admin > Models if you want Open WebUI's own web search or code interpreter.

Standard library only. Exit 0 when every Mference model is configured, 2 when
Open WebUI cannot be reached or refuses the token.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

OWNER = "mference"


def request(base, path, token=None, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read()
    return json.loads(body) if body else None


def sign_in(base, token):
    """Return an admin token: the one given, else the no-auth session Open WebUI
    hands out when WEBUI_AUTH is false (the same call the frontend makes)."""
    if token:
        return token
    try:
        result = request(base, "/api/v1/auths/signin", payload={"email": "", "password": ""})
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"error: Open WebUI refused an anonymous sign-in ({error.code}); authentication is on. "
            "Pass --token <admin API token> or set OPEN_WEBUI_TOKEN."
        )
    if not result or not result.get("token"):
        raise SystemExit("error: Open WebUI returned no session token")
    if result.get("role") != "admin":
        raise SystemExit(f"error: signed in as role {result.get('role')!r}; an admin token is required")
    return result["token"]


def mference_models(base, token):
    served = request(base, "/api/models", token) or {}
    models = []
    for item in served.get("data", []):
        owners = {item.get("owned_by"), (item.get("openai") or {}).get("owned_by")}
        if OWNER in owners:
            models.append(item)
    return models


def existing_entry(base, token, model_id):
    try:
        return request(base, "/api/v1/models/model?id=" + urllib.parse.quote(model_id, safe=""), token)
    except urllib.error.HTTPError as error:
        if error.code in (401, 404):
            return None
        raise


def configure(base, token, item, dry_run):
    model_id = item["id"]
    name = item.get("name") or model_id
    entry = existing_entry(base, token, model_id)
    if entry:
        meta = dict(entry.get("meta") or {})
        capabilities = dict(meta.get("capabilities") or {})
        if capabilities.get("builtin_tools") is False:
            return "already off"
        capabilities["builtin_tools"] = False
        meta["capabilities"] = capabilities
        form = {
            "id": model_id,
            "base_model_id": entry.get("base_model_id"),
            "name": entry.get("name") or name,
            "meta": meta,
            "params": entry.get("params") or {},
            "is_active": entry.get("is_active", True),
        }
        if not dry_run:
            request(base, "/api/v1/models/model/update?id=" + urllib.parse.quote(model_id, safe=""), token, form)
        return "updated"
    form = {
        "id": model_id,
        "base_model_id": None,
        "name": name,
        "meta": {"capabilities": {"builtin_tools": False}},
        "params": {},
        "is_active": True,
    }
    if not dry_run:
        request(base, "/api/v1/models/create", token, form)
    return "created"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--webui", default="http://127.0.0.1:3000", help="Open WebUI base URL")
    parser.add_argument("--token", default=os.environ.get("OPEN_WEBUI_TOKEN"), help="admin API token (optional when WEBUI_AUTH=false)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = parser.parse_args()
    base = args.webui.rstrip("/")

    try:
        token = sign_in(base, args.token)
        models = mference_models(base, token)
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(f"error: cannot reach Open WebUI at {base}: {error}")

    if not models:
        print(f"[configure-models] no Mference models advertised by {base} yet", file=sys.stderr)
        return 0

    failures = 0
    for item in models:
        try:
            outcome = configure(base, token, item, args.dry_run)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:200]
            print(f"[configure-models] {item['id']}: FAILED {error.code} {detail}", file=sys.stderr)
            failures += 1
            continue
        verb = "would be " + outcome.replace("already off", "left as is") if args.dry_run else outcome
        print(f"[configure-models] {item['id']}: builtin tools off ({verb})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
