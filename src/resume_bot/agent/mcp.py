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
