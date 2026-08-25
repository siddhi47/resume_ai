import asyncio
import os

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent

# Compatibility shim: this version of langgraph-checkpoint-sqlite's AsyncSqliteSaver.setup()
# calls conn.is_alive(), a method aiosqlite removed. The connection is always freshly opened
# and used within the same `async with` block here, so "always alive" is a safe stand-in.
if not hasattr(aiosqlite.Connection, "is_alive"):
    aiosqlite.Connection.is_alive = lambda self: True

from src.resume_bot.agent.gatekeeper import REJECTION_MESSAGE, is_in_scope
from src.resume_bot.agent.mcp import load_github_tools
from src.resume_bot.agent.prompts import get_system_prompt
from src.resume_bot.agent.state import AgentState
from src.resume_bot.agent.tools import LOCAL_TOOLS

CHECKPOINT_DB_PATH = os.path.join("data", "agent_checkpoints.sqlite")

_llm = None


def _get_llm():
    # Built lazily, not at import time: app.py imports this module before calling its own
    # load_dotenv(), so OPENAI_API_KEY wouldn't be set yet if this ran at module load.
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    return _llm


def _ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def _build_agent(tools, checkpointer):
    return create_react_agent(
        model=_get_llm(),
        tools=tools,
        prompt=get_system_prompt(),
        state_schema=AgentState,
        checkpointer=checkpointer,
    )


async def _invoke(message: str, thread_id: str, user_id: str) -> dict:
    _ensure_data_dir()
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    if not await is_in_scope(message):
        # Rejected before any tool-calling agent runs: cheap, and keeps off-topic requests
        # from ever reaching the real (more expensive, tool-equipped) model.
        async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
            agent = _build_agent(LOCAL_TOOLS, checkpointer)
            ai_message = AIMessage(content=REJECTION_MESSAGE)
            await agent.aupdate_state(
                config,
                {"messages": [HumanMessage(content=message), ai_message]},
                as_node="agent",
            )
            return {"messages": [ai_message]}

    github_tools = await load_github_tools()
    tools = LOCAL_TOOLS + github_tools

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        agent = _build_agent(tools, checkpointer)
        return await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )


def invoke_agent_sync(message: str, thread_id: str, user_id: str) -> dict:
    return asyncio.run(_invoke(message, thread_id, user_id))


def get_conversation_history(thread_id: str):
    """Read prior turns for GET rendering. Uses only LOCAL_TOOLS (no MCP/network) so viewing
    history never fails because of a GitHub token/connectivity issue."""
    _ensure_data_dir()
    with SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        agent = _build_agent(LOCAL_TOOLS, checkpointer)
        snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})

    values = snapshot.values if snapshot else {}
    messages = values.get("messages", [])

    history = []
    for m in messages:
        if m.type == "human":
            history.append({"role": "user", "content": m.content})
        elif m.type == "ai" and m.content:
            history.append({"role": "assistant", "content": m.content})

    return history, values.get("last_artifact_path"), values.get("last_artifact_label")
