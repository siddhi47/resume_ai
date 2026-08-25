import os

from langchain_mcp_adapters.client import MultiServerMCPClient

WANTED_GITHUB_TOOLS = {"search_repositories", "get_file_contents", "search_code"}

# Two ways to reach GitHub's MCP server:
#   "hosted"      - GitHub's remote server at api.githubcopilot.com. No extra infra, but may
#                    require the account to have a GitHub Copilot plan enabled to authenticate.
#   "self_hosted" - the official github-mcp-server binary, run as a subprocess over stdio.
#                    Install it into the Docker image the same way tectonic is installed
#                    (see Dockerfile) if you go this route.
GITHUB_MCP_MODE = os.getenv("GITHUB_MCP_MODE", "hosted")


_resolved_username = None


def resolve_github_username() -> str:
    """The GitHub username, from GITHUB_USERNAME or — failing that — from the token itself.

    A valid token already identifies its owner, so requiring a separate env var just creates a
    way to half-configure the app: tools load fine but the agent is told GitHub is unavailable
    and never looks anything up. Falling back to /user removes that failure mode entirely.
    """
    global _resolved_username
    username = os.environ.get("GITHUB_USERNAME", "").strip()
    if username:
        return username
    if _resolved_username is not None:
        return _resolved_username

    _resolved_username = ""
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
    if token:
        try:
            import requests

            resp = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                _resolved_username = resp.json().get("login") or ""
                if _resolved_username:
                    print(
                        f"[agent] GITHUB_USERNAME not set; resolved '{_resolved_username}' "
                        "from the token."
                    )
        except Exception as exc:
            print(f"[agent] Could not resolve GitHub username from token: {exc}")
    return _resolved_username


def _github_connection() -> dict:
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        return None

    if GITHUB_MCP_MODE == "self_hosted":
        return {
            "github": {
                "transport": "stdio",
                "command": "github-mcp-server",
                "args": ["stdio"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": token},
            }
        }

    return {
        "github": {
            "transport": "streamable_http",
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": f"Bearer {token}"},
        }
    }


async def load_github_tools() -> list:
    """Load GitHub MCP tools if a token is configured. Never raises: on any connection/auth
    failure (e.g. the hosted endpoint rejecting the token), logs a warning and returns []
    so the rest of the agent (cover letter, resume tailoring, Q&A) still works."""
    connection = _github_connection()
    if connection is None:
        return []

    try:
        client = MultiServerMCPClient(connection)
        all_tools = await client.get_tools()
        return [t for t in all_tools if t.name in WANTED_GITHUB_TOOLS]
    except Exception as exc:
        print(f"[agent] GitHub MCP tool loading failed, continuing without it: {exc}")
        return []
