import asyncio
import os

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import create_react_agent

# Compatibility shim: this version of langgraph-checkpoint-sqlite's AsyncSqliteSaver.setup()
# calls conn.is_alive(), a method aiosqlite removed. The connection is always freshly opened
# and used within the same `async with` block here, so "always alive" is a safe stand-in.
if not hasattr(aiosqlite.Connection, "is_alive"):
    aiosqlite.Connection.is_alive = lambda self: True

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


def _healing_tool_message(call_id):
    return ToolMessage(
        content=(
            "(This tool call was interrupted before it finished, e.g. by a server restart, "
            "and its result was lost. Retry it if the answer still needs it.)"
        ),
        tool_call_id=call_id,
    )


async def _heal_incomplete_tool_calls(agent, config):
    """If a previous turn was interrupted (crash, worker restart, deploy) after the model
    requested tool calls but before they finished, the checkpoint is left with an AIMessage
    whose tool_calls have no matching ToolMessage. Every future turn on that thread then fails
    _validate_chat_history before the model is ever called, permanently bricking the
    conversation.

    It's not enough to just append a placeholder ToolMessage anywhere in the state — LangGraph's
    own validation only checks that a matching ToolMessage exists *somewhere*, but OpenAI's API
    requires it to *immediately follow* the assistant message that requested it. Appending to the
    end of a long-since-continued thread satisfies the former but not the latter (and fails with
    an OpenAI 400 instead). So instead, rebuild the full message list with each placeholder
    spliced in right after the AIMessage it resolves, and overwrite state with it wholesale via
    the RemoveMessage(id=REMOVE_ALL_MESSAGES) sentinel.
    """
    snapshot = await agent.aget_state(config)
    messages = snapshot.values.get("messages", []) if snapshot else []
    if not messages:
        return

    resolved_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    has_unresolved = any(
        call["id"] not in resolved_ids
        for m in messages
        if isinstance(m, AIMessage)
        for call in (m.tool_calls or [])
    )
    if not has_unresolved:
        return

    rebuilt = []
    for m in messages:
        rebuilt.append(m)
        if isinstance(m, AIMessage):
            for call in m.tool_calls or []:
                if call["id"] not in resolved_ids:
                    rebuilt.append(_healing_tool_message(call["id"]))

    await agent.aupdate_state(
        config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + rebuilt},
        as_node="tools",
    )


async def _invoke(message: str, thread_id: str, user_id: str) -> dict:
    _ensure_data_dir()
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    github_tools = await load_github_tools()
    tools = LOCAL_TOOLS + github_tools

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        agent = _build_agent(tools, checkpointer)
        await _heal_incomplete_tool_calls(agent, config)
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
