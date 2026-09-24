"""Diagnose GitHub MCP connectivity, one layer at a time.

Run it wherever the app actually runs — inside the container, not on your laptop:

    docker compose exec web python scripts/check_github_mcp.py
    # or, if the service has another name:
    docker exec -it flask_resume_chatbot python scripts/check_github_mcp.py

Each check isolates a different failure, so the first FAIL tells you what to fix.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))

OK = "PASS"
NO = "FAIL"


def line(status, label, detail=""):
    print(f"[{status}] {label}" + (f"\n       {detail}" if detail else ""))


def check_env():
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    username = os.environ.get("GITHUB_USERNAME", "")
    mode = os.environ.get("GITHUB_MCP_MODE", "hosted")

    if not token:
        line(NO, "GITHUB_PERSONAL_ACCESS_TOKEN is not set", "Add it to .env, then RESTART the container.")
        return None, None
    line(OK, f"GITHUB_PERSONAL_ACCESS_TOKEN is set ({token[:12]}...{token[-4:]}, {len(token)} chars)")

    if not username:
        line(NO, "GITHUB_USERNAME is not set", "Without it the agent skips GitHub entirely. Add it to .env.")
    else:
        line(OK, f"GITHUB_USERNAME = {username}")

    line(OK, f"GITHUB_MCP_MODE = {mode}")
    return token, username


def check_token(token):
    """Validates the token itself and general outbound network, independent of MCP."""
    import requests

    try:
        resp = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except Exception as exc:
        line(NO, "Cannot reach api.github.com at all", f"{type(exc).__name__}: {exc}\n       Network/DNS/firewall problem on this host.")
        return False

    if resp.status_code == 200:
        line(OK, f"Token authenticates to GitHub as '{resp.json().get('login')}'")
        return True
    if resp.status_code == 401:
        line(NO, "GitHub rejected the token (401)", "It is expired or revoked. Generate a new PAT.")
    else:
        line(NO, f"GitHub returned HTTP {resp.status_code}", resp.text[:200])
    return False


def check_mcp_endpoint(token):
    """The hosted MCP server is a different host from api.github.com and can fail alone."""
    import requests

    try:
        resp = requests.post(
            "https://api.githubcopilot.com/mcp/",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            timeout=20,
        )
    except Exception as exc:
        line(NO, "Cannot reach api.githubcopilot.com", f"{type(exc).__name__}: {exc}\n       Egress to this host is likely blocked (proxy/firewall).")
        return False

    if resp.status_code in (200, 202):
        line(OK, f"MCP endpoint reachable (HTTP {resp.status_code})")
        return True
    if resp.status_code in (401, 403):
        line(NO, f"MCP endpoint rejected the token (HTTP {resp.status_code})", f"{resp.text[:200]}\n       The PAT may lack scopes, or this account can't use the hosted server.\n       Fallback: set GITHUB_MCP_MODE=self_hosted.")
        return False
    line(OK, f"MCP endpoint responded HTTP {resp.status_code} (not an auth failure)")
    return True


async def check_tool_loading():
    from src.resume_bot.agent.mcp import load_github_tools

    tools = await load_github_tools()
    if not tools:
        line(NO, "load_github_tools() returned no tools", "See the exception printed above this line, if any.")
        return None
    line(OK, f"Loaded MCP tools: {', '.join(t.name for t in tools)}")
    return tools


async def check_real_search():
    from src.resume_bot.agent.planned_graph import search_github

    result = await search_github({})
    text = result.get("github_repos", "")
    if text.startswith(("GitHub lookup failed", "GitHub lookup unavailable", "GitHub lookup is not configured")):
        line(NO, "Repo listing failed", text[:300])
        return
    count = len([l for l in text.splitlines() if l.startswith("- ")])
    line(OK, f"Repo listing works — {count} repositories returned")
    for l in text.splitlines()[:3]:
        print(f"         {l[:100]}")


async def main():
    print("=" * 70)
    print("GitHub MCP diagnostic")
    print("=" * 70)
    print(f"cwd: {os.getcwd()}")
    print(f".env present here: {os.path.exists('.env')}")
    print("-" * 70)

    token, _username = check_env()
    if not token:
        print("\nStop: nothing else can work without the token.")
        return

    print("-" * 70)
    if not check_token(token):
        print("\nStop: fix the token/network before looking at MCP.")
        return

    print("-" * 70)
    check_mcp_endpoint(token)

    print("-" * 70)
    if await check_tool_loading():
        print("-" * 70)
        await check_real_search()

    print("=" * 70)


asyncio.run(main())

