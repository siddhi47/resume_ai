from typing import Optional

from langgraph.prebuilt.chat_agent_executor import AgentState as _PrebuiltAgentState


class AgentState(_PrebuiltAgentState):
    last_artifact_path: Optional[str]
    last_artifact_label: Optional[str]
