"""A deterministic, node-based alternative to the ReAct agent in graph.py.

graph.py hands the LLM a set of tools and lets it decide which to call and in what
order, which means "look up GitHub before tailoring" is only ever a suggestion the
model can skip — the cause of several real failures ("I didn't find any AI/ML repos"
when 68 existed).

Here the gathering steps are unconditional graph nodes, so they always run, and the
two review points are real interrupt() pauses rather than prompt instructions, so
they can't be skipped either.

    START -> read_intent -+-> answer_directly -> END
                          |
                          +-> fetch_jd -> [search_github, read_resume_tex,
                          |               check_uploaded_resume, extract_jd_insights]
                          |                        |
                          +------------------------+-> propose_plan
                                                        |
                                                   await_plan_approval
                                                   |              |
                                            (changes)          (approved)
                                                   |              |
                                                   +-> propose_plan
                                                                  v
                                                             draft_reply
                                                                  |
                                                        await_draft_approval
                                                        |              |
                                                 (changes)          (approved)
                                                        |              |
                                                        +-> draft_reply
                                                                       v
                                                                  finalize -> END
"""

import json
import os
import re
from typing import Annotated, Optional, TypedDict

import aiosqlite
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

# Same compatibility shim as graph.py: this version of langgraph-checkpoint-sqlite's
# AsyncSqliteSaver.setup() calls conn.is_alive(), a method aiosqlite removed.
if not hasattr(aiosqlite.Connection, "is_alive"):
    aiosqlite.Connection.is_alive = lambda self: True

from src.resume_bot.agent.generation import (
    HREF_PATTERN,
    URL_PATTERN,
    build_cover_letter,
    build_tailored_resume,
    resume_template_path,
)
from src.resume_bot.agent.mcp import load_github_tools, resolve_github_username
from src.resume_bot.agent.tools import QA_TOOLS, fetch_job_posting_text
from src.resume_bot.shared import get_job_context

CHECKPOINT_DB_PATH = os.path.join("data", "planned_checkpoints.sqlite")

# A pasted block this long is almost certainly a job description rather than a question.
# Deterministic on purpose: the old LLM-based gatekeeper kept misclassifying pasted JDs.
PASTED_JD_MIN_CHARS = 400

GATHER_NODES = [
    "search_github",
    "read_resume_tex",
    "check_uploaded_resume",
    "extract_jd_insights",
]


class PlannedState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: Optional[str]
    doc_types: Optional[list]
    pending_url: Optional[str]
    job_description: Optional[str]
    company: Optional[str]
    jd_insights: Optional[str]
    github_repos: Optional[str]
    existing_projects: Optional[str]
    resume_template_present: Optional[bool]
    uploaded_resume_summary: Optional[str]
    contact_info: Optional[str]
    plan: Optional[str]
    plan_feedback: Optional[str]
    draft_feedback: Optional[str]
    artifacts: Optional[list]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _llm(model="gpt-4o-mini", temperature=0):
    # Built per call, not at import: app.py imports this module before load_dotenv() runs.
    return ChatOpenAI(model=model, temperature=temperature)


def _tool_calling_llm(model="gpt-4o", temperature=0.4):
    # The legacy community ChatOpenAI above has no bind_tools, so tool-using nodes need
    # the langchain_openai client (the same one graph.py's ReAct agent uses).
    from langchain_openai import ChatOpenAI as ToolChatOpenAI

    return ToolChatOpenAI(model=model, temperature=temperature)


def _run(template, model="gpt-4o-mini", temperature=0, **kwargs):
    prompt = PromptTemplate(input_variables=list(kwargs), template=template)
    return LLMChain(llm=_llm(model, temperature), prompt=prompt).run(**kwargs).strip()


def _last_human_text(state):
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


_AFFIRMATIVE = {
    "yes", "y", "yeah", "yep", "ya", "ok", "okay", "k", "sure", "go", "go ahead",
    "do it", "proceed", "continue", "looks good", "lgtm", "sounds good", "perfect",
    "great", "approve", "approved", "confirm", "confirmed", "yes please", "good",
    "fine", "correct", "right", "ship it", "send it",
}

_APPROVAL_PROMPT = """A proposal was shown to a user, who replied with the text below.

Did they approve it as-is, or are they asking for a change?
Answer with exactly one word: APPROVE or CHANGES.
If they approve but also add a new instruction or correction, answer CHANGES.

User reply:
{reply}
"""


def _is_approval(text):
    """Deterministic match on common affirmatives first, LLM only for genuinely
    ambiguous replies — the same lesson the removed gatekeeper taught: don't hand an
    LLM a judgement that a string comparison already answers correctly."""
    normalized = re.sub(r"[^a-z ]", "", (text or "").lower()).strip()
    if not normalized:
        return False
    if normalized in _AFFIRMATIVE:
        return True
    if len(normalized.split()) > 6:
        return _run(_APPROVAL_PROMPT, reply=text).upper().startswith("APPROVE")
    for phrase in _AFFIRMATIVE:
        if normalized.startswith(phrase + " ") or normalized == phrase:
            return True
    return _run(_APPROVAL_PROMPT, reply=text).upper().startswith("APPROVE")


def _doc_types(intent):
    if intent == "both":
        return ["resume", "cover_letter"]
    if intent in ("resume", "cover_letter"):
        return [intent]
    return []


def _describe(doc_types):
    names = {"resume": "tailored resume", "cover_letter": "cover letter"}
    return " and ".join(names[d] for d in doc_types)


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------

_INTENT_PROMPT = """Classify what the user wants. Answer with exactly one word:

- resume       : they want their resume tailored to a job
- cover_letter : they want a cover letter
- both         : they explicitly want both documents
- revise       : they are giving feedback on a document that was just produced for them
- generic      : anything else (a question, chit-chat, a request unrelated to producing a document)

Context:
- A job description is already on file for this conversation: {has_prior_jd}
- A document was already produced for them in this conversation: {has_artifact}

If they are asking for a document for a NEW job, that is resume/cover_letter/both, not revise.
Only answer "revise" if a document already exists AND their message reads as feedback on it.

User message:
{message}
"""


def read_intent(state):
    message = _last_human_text(state)
    has_prior_jd = bool(state.get("job_description"))
    has_artifact = bool(state.get("artifacts"))

    intent = _run(
        _INTENT_PROMPT,
        message=message[:4000],
        has_prior_jd="yes" if has_prior_jd else "no",
        has_artifact="yes" if has_artifact else "no",
    ).strip().lower()
    intent = re.sub(r"[^a-z_]", "", intent)
    if intent not in ("resume", "cover_letter", "both", "revise", "generic"):
        intent = "generic"

    updates = {"intent": intent}
    doc_types = _doc_types(intent)
    if doc_types:
        updates["doc_types"] = doc_types

    # Deterministic: a URL or a long pasted block means a (possibly new) job posting.
    url_match = URL_PATTERN.search(message)
    if url_match:
        updates["pending_url"] = url_match.group(0)
        updates["job_description"] = None
        updates["jd_insights"] = None
    elif len(message) >= PASTED_JD_MIN_CHARS and intent != "revise":
        updates["job_description"] = message
        updates["jd_insights"] = None
    return updates


def route_after_intent(state):
    intent = state.get("intent")
    if intent == "generic":
        return "answer_directly"
    if intent == "revise":
        return "draft_reply" if state.get("artifacts") else "answer_directly"
    if state.get("pending_url"):
        return "fetch_jd"
    if not state.get("job_description"):
        return "answer_directly"
    # Same job, different document (or a re-run): everything gathered is still valid.
    if state.get("jd_insights"):
        return "propose_plan"
    return list(GATHER_NODES)


def fetch_jd(state):
    url = state.get("pending_url")
    try:
        text = fetch_job_posting_text.invoke({"url": url})
    except Exception as exc:
        return {
            "pending_url": None,
            "messages": [
                AIMessage(
                    content=(
                        f"I couldn't fetch that job posting ({exc}). Paste the job "
                        "description text directly and I'll work from that."
                    )
                )
            ],
        }
    return {"job_description": text, "pending_url": None}


def route_after_fetch(state):
    return list(GATHER_NODES) if state.get("job_description") else END


def _summarize_repo_payload(raw):
    """The MCP tool hands back a JSON blob; reduce it to the fields the planner needs."""
    data = raw
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            return data[:6000]
    if isinstance(data, dict):
        data = data.get("items") or data.get("repositories") or []
    if not isinstance(data, list):
        return str(data)[:6000]

    lines = []
    for repo in data:
        if not isinstance(repo, dict):
            continue
        name = repo.get("name") or repo.get("full_name") or "?"
        url = repo.get("html_url") or repo.get("url") or ""
        desc = (repo.get("description") or "").strip() or "no description"
        language = repo.get("language") or ""
        topics = ", ".join(repo.get("topics") or [])
        extra = " | ".join(x for x in (language, topics) if x)
        lines.append(f"- {name} ({url}): {desc}" + (f" [{extra}]" if extra else ""))
    return "\n".join(lines) if lines else "No repositories returned."


async def search_github(state):
    username = resolve_github_username()
    if not username:
        return {
            "github_repos": (
                "GitHub lookup is not configured — no GITHUB_USERNAME, and the username "
                "could not be resolved from GITHUB_PERSONAL_ACCESS_TOKEN either."
            )
        }
    try:
        tools = await load_github_tools()
        search = next((t for t in tools if t.name == "search_repositories"), None)
        if search is None:
            return {
                "github_repos": (
                    "GitHub lookup unavailable — the MCP tools did not load "
                    "(check GITHUB_PERSONAL_ACCESS_TOKEN)."
                )
            }
        # Deliberately unscoped by keyword: GitHub search is literal, so filtering by an
        # abstract category ("machine learning") silently drops relevant repos. Pull the
        # whole list and let propose_plan judge relevance.
        raw = await search.ainvoke({"query": f"user:{username}", "perPage": 100})
        return {"github_repos": _summarize_repo_payload(raw)}
    except Exception as exc:
        return {"github_repos": f"GitHub lookup failed: {exc}"}


def read_resume_tex(state, config):
    user_id = config["configurable"]["user_id"]
    path = resume_template_path(user_id)
    if not os.path.exists(path):
        return {"resume_template_present": False, "existing_projects": ""}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        tex = f.read()
    return {
        "resume_template_present": True,
        "existing_projects": "\n".join(sorted(set(HREF_PATTERN.findall(tex)))),
    }


def check_uploaded_resume(state):
    return {"uploaded_resume_summary": (state.get("uploaded_resume_summary") or "").strip()}


_JD_INSIGHTS_PROMPT = """From the job description below, extract only what it actually states:

- Role title and company
- The 5-8 most important required skills / technologies
- The core responsibilities, in one or two sentences

Be concise and factual. Do not infer or embellish.

Job Description:
{job_description}
"""


def extract_jd_insights(state):
    jd = state.get("job_description") or ""
    summary, company = get_job_context(jd)
    insights = _run(_JD_INSIGHTS_PROMPT, job_description=summary[:12000])
    return {"jd_insights": insights, "company": company}


_PLAN_PROMPT = """You are preparing to produce a {doc_description} for a job application.
Write a SHORT plan for the user to approve. Do NOT write the document itself.

What the job needs:
{jd_insights}

The candidate's real GitHub repositories:
{github_repos}

Project URLs already present in the candidate's resume:
{existing_projects}

Feedback on your previous plan (if any): {plan_feedback}

Write 4-8 short lines covering:
- One line on what this role is really looking for.
- Which specific GitHub repositories you would add as NEW resume project entries. Name each
  one and include its full URL exactly as given above. Only choose from the list above. If
  none are a genuine fit, say so plainly instead of stretching.
- Which parts of the existing resume you would re-emphasise.
- A closing line asking whether this looks right or they want changes.

Do not invent repositories, skills, or experience. No markdown headings.
"""


def propose_plan(state):
    doc_types = state.get("doc_types") or ["resume"]
    plan = _run(
        _PLAN_PROMPT,
        model="gpt-4o",
        doc_description=_describe(doc_types),
        jd_insights=state.get("jd_insights") or "(not extracted)",
        github_repos=state.get("github_repos") or "(none found)",
        existing_projects=state.get("existing_projects") or "(none)",
        plan_feedback=state.get("plan_feedback") or "None.",
    )
    return {"plan": plan, "plan_feedback": None, "messages": [AIMessage(content=plan)]}


def await_plan_approval(state):
    decision = interrupt({"kind": "plan_approval"})
    text = (decision or "").strip()
    return {
        "messages": [HumanMessage(content=text)],
        "plan_feedback": None if _is_approval(text) else text,
    }


def route_after_plan(state):
    return "propose_plan" if state.get("plan_feedback") else "draft_reply"


def draft_reply(state, config):
    user_id = config["configurable"]["user_id"]
    doc_types = state.get("doc_types") or ["resume"]
    jd = state.get("job_description") or ""
    feedback = state.get("draft_feedback") or "None."
    # The approved plan is the source of truth for which projects may be added — the
    # generators extract allowed URLs from this text, so nothing outside the plan can
    # sneak into the resume as a new entry.
    approved_projects = state.get("plan") or ""

    artifacts = []
    notes = []
    for doc_type in doc_types:
        if doc_type == "resume":
            result = build_tailored_resume(
                user_id=user_id,
                job_description=jd,
                relevant_projects=approved_projects,
                revision_feedback=feedback,
            )
            if result.get("error"):
                notes.append(result["error"])
                continue
            notes.append(result["message"])
        else:
            result = build_cover_letter(
                user_id=user_id,
                job_description=jd,
                resume_summary=state.get("uploaded_resume_summary") or "",
                contact_info=state.get("contact_info") or "",
                relevant_projects=approved_projects,
                revision_feedback=feedback,
            )
            issues = result["remaining_issues"]
            notes.append(
                "Cover letter drafted."
                + (
                    f" (validation flagged: {'; '.join(issues)} — worth a read before sending)"
                    if issues
                    else ""
                )
            )
        artifacts.append({"path": result["filename"], "label": result["label"]})

    summary = "\n".join(notes) + "\n\nHave a look — want any changes, or is this good to finalise?"
    return {
        "artifacts": artifacts,
        "draft_feedback": None,
        "messages": [AIMessage(content=summary)],
    }


def await_draft_approval(state):
    decision = interrupt({"kind": "draft_approval"})
    text = (decision or "").strip()
    return {
        "messages": [HumanMessage(content=text)],
        "draft_feedback": None if _is_approval(text) else text,
    }


def route_after_draft(state):
    return "draft_reply" if state.get("draft_feedback") else "finalize"


def finalize(state):
    doc_types = state.get("doc_types") or ["resume"]
    other = "cover letter" if "resume" in doc_types and "cover_letter" not in doc_types else None
    extra = (
        f" If you want a {other} for this same job, just ask — I'll reuse everything I already gathered."
        if other
        else ""
    )
    return {
        "messages": [
            AIMessage(content=f"Finalised. The download link is below.{extra}")
        ]
    }


@tool
async def list_my_github_repos() -> str:
    """List ALL of the user's GitHub repositories, with name, URL, description and language.

    Always returns their complete repo list. Judge relevance yourself by reading the names and
    descriptions — a repo about ECG classification or medical imaging IS a machine-learning
    project even if it never says so. Never conclude a repo doesn't exist without calling this.
    """
    result = await search_github({})
    return result["github_repos"]


@tool
def get_resume_details(config: RunnableConfig) -> str:
    """Return the full text of the user's resume: experience, employers, dates, education,
    publications, projects, and skills. Use this for any question about their background."""
    user_id = config["configurable"]["user_id"]
    path = resume_template_path(user_id)
    if not os.path.exists(path):
        return f"No resume template found at {path}."
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()[:40000]


# Deliberately NOT the raw MCP search tools: letting the model compose its own GitHub query
# is what produced "I didn't find any AI/ML repositories" when 68 existed, since GitHub's
# search matches literal keywords. These wrappers take no query at all, so that failure mode
# is unreachable — the model only chooses *whether* to look, never *how*.
_QA_TOOLS = QA_TOOLS + [list_my_github_repos, get_resume_details]

_QA_SYSTEM_PROMPT = """You are Siddhi's job-application assistant, answering a question in the
middle of an ongoing conversation.

Your tools:
- get_resume_details — the user's actual resume. Use it for ANY question about their
  background: experience, employers, dates, education, publications, projects, skills.
  Publications and past jobs live here, not on GitHub.
- list_my_github_repos — their complete GitHub repository list. Use it for any question
  about their code or repos. It always returns everything, so decide relevance yourself by
  reading the descriptions. Never say a repo doesn't exist without calling this first.
- fetch_job_posting_text — pull a job posting from a URL.

Call a tool whenever it would ground your answer in fact rather than a guess. Never invent a
repository, employer, date, publication, or metric.

You cannot produce a resume or cover letter yourself. If the user is asking for one, tell them
you'll start on it and that you'll show them a plan first — do not attempt to write the document
in your reply.

Be direct and conversational. Don't restate the question before answering it. Never write
bracketed placeholder text; if a detail isn't available, leave it out.
"""


async def answer_directly(state, config):
    message = _last_human_text(state)
    if state.get("intent") in ("resume", "cover_letter", "both") and not state.get(
        "job_description"
    ):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Happy to do that — paste the job posting URL, or the job "
                        "description text, and I'll get started."
                    )
                )
            ]
        }
    if state.get("intent") == "revise" and not state.get("artifacts"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "There's nothing drafted yet in this conversation. Send me a job "
                        "posting and I'll put a plan together first."
                    )
                )
            ]
        }
    # A ReAct sub-agent, deliberately: open-ended questions are exactly the case where
    # letting the model choose its own tools beats a fixed pipeline. It gets _QA_TOOLS
    # only — no document generators — so it can research freely but can't route around
    # the approval gates that the rest of this graph exists to enforce.
    agent = create_react_agent(
        model=_tool_calling_llm(),
        tools=_QA_TOOLS,
        prompt=_QA_SYSTEM_PROMPT,
    )
    result = await agent.ainvoke({"messages": state.get("messages", [])}, config=config)

    answer = ""
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            answer = m.content
            break
    # Only the final answer joins this graph's state — the sub-agent's internal tool
    # traffic stays inside it, so a mid-flight failure can't leave a dangling tool call
    # in the checkpoint (the bug that bricked threads on the /agent route).
    return {"messages": [AIMessage(content=answer or "(no response)")]}


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------

def build_planned_graph(checkpointer):
    builder = StateGraph(PlannedState)

    builder.add_node("read_intent", read_intent)
    builder.add_node("answer_directly", answer_directly)
    builder.add_node("fetch_jd", fetch_jd)
    builder.add_node("search_github", search_github)
    builder.add_node("read_resume_tex", read_resume_tex)
    builder.add_node("check_uploaded_resume", check_uploaded_resume)
    builder.add_node("extract_jd_insights", extract_jd_insights)
    builder.add_node("propose_plan", propose_plan)
    builder.add_node("await_plan_approval", await_plan_approval)
    builder.add_node("draft_reply", draft_reply)
    builder.add_node("await_draft_approval", await_draft_approval)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "read_intent")
    builder.add_conditional_edges(
        "read_intent",
        route_after_intent,
        ["answer_directly", "fetch_jd", "draft_reply", "propose_plan"] + GATHER_NODES,
    )
    builder.add_conditional_edges("fetch_jd", route_after_fetch, GATHER_NODES + [END])
    for node in GATHER_NODES:
        builder.add_edge(node, "propose_plan")

    builder.add_edge("answer_directly", END)
    builder.add_edge("propose_plan", "await_plan_approval")
    builder.add_conditional_edges(
        "await_plan_approval", route_after_plan, ["draft_reply", "propose_plan"]
    )
    builder.add_edge("draft_reply", "await_draft_approval")
    builder.add_conditional_edges(
        "await_draft_approval", route_after_draft, ["finalize", "draft_reply"]
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def _ensure_data_dir():
    os.makedirs("data", exist_ok=True)


async def _invoke(message, thread_id, user_id, resume_summary="", contact_info=""):
    _ensure_data_dir()
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        graph = build_planned_graph(checkpointer)
        snapshot = await graph.aget_state(config)

        # A thread paused at an interrupt is waiting for an answer to a specific question,
        # so this message resumes it rather than starting a fresh turn.
        if snapshot and snapshot.next:
            return await graph.ainvoke(Command(resume=message), config=config)

        return await graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "uploaded_resume_summary": resume_summary,
                "contact_info": contact_info,
            },
            config=config,
        )


def invoke_planned_sync(message, thread_id, user_id, resume_summary="", contact_info=""):
    import asyncio

    return asyncio.run(_invoke(message, thread_id, user_id, resume_summary, contact_info))


def get_planned_history(thread_id):
    """Read prior turns for GET rendering, without touching MCP/network."""
    import asyncio

    async def _read():
        _ensure_data_dir()
        async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
            graph = build_planned_graph(checkpointer)
            return await graph.aget_state({"configurable": {"thread_id": thread_id}})

    snapshot = asyncio.run(_read())
    values = snapshot.values if snapshot else {}

    history = []
    for m in values.get("messages", []):
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            history.append({"role": "assistant", "content": m.content})

    return history, values.get("artifacts") or []
