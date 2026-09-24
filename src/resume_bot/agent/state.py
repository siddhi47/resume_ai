from typing import Annotated, Optional

from langgraph.prebuilt.chat_agent_executor import AgentState as _PrebuiltAgentState


def _latest_artifact(current: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """Merge concurrent writes to an artifact key, keeping the most recent non-empty one.

    Both build_cover_letter and build_tailored_resume set these keys. When the model
    emits those two tool calls in the *same* step - which some models do far more
    readily than others - LangGraph sees two updates to one key in a single step and,
    with no reducer declared, raises InvalidUpdateError. That poisons the thread: even
    reading the conversation back fails afterwards, so the whole /agent page 500s.

    Last-write-wins is the right semantic here, since these keys only track "which file
    to offer the user for download right now".
    """
    return incoming if incoming is not None else current


class AgentState(_PrebuiltAgentState):
    last_artifact_path: Annotated[Optional[str], _latest_artifact]
    last_artifact_label: Annotated[Optional[str], _latest_artifact]
