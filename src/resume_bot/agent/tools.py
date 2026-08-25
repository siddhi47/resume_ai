from typing import Annotated

import requests
from bs4 import BeautifulSoup
from flask import session
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.resume_bot.shared import (
    GENERIC_QA_PROMPT_TEMPLATE,
    get_job_context,
)
from src.resume_bot.agent.generation import (
    build_cover_letter,
    build_tailored_resume,
)


@tool
def fetch_job_posting_text(url: str) -> str:
    """Fetch a job posting from a URL and return its extracted plain-text content."""
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (resume-bot/1.0)"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:20000]


@tool
def get_resume_context() -> str:
    """Return the cached resume summary and contact info for this user, if a resume has been uploaded via the /chat page."""
    resume_summary = session.get("resume_summary")
    extracted_info = session.get("extracted_info")
    if not resume_summary:
        return "No resume on file. Ask the user to upload a PDF resume via the /chat page first."
    return f"CONTACT INFO:\n{extracted_info}\n\nRESUME:\n{resume_summary}"


@tool
def answer_application_question(
    user_question: str, job_description: str
) -> str:
    """Answer a job-application question as the applicant, using the resume on file and the given job description text."""
    resume_summary = session.get("resume_summary") or ""
    job_summary, _company = get_job_context(job_description)

    llm = ChatOpenAI(temperature=0.5)
    prompt = PromptTemplate(
        input_variables=["resume_summary", "job_description", "user_question"],
        template=GENERIC_QA_PROMPT_TEMPLATE,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(
        resume_summary=resume_summary,
        job_description=job_summary,
        user_question=user_question,
    )


@tool
def generate_cover_letter(
    job_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    relevant_projects: str = "",
) -> Command:
    """Draft a tailored cover letter for the given job description text and save it as a downloadable PDF.

    If you already looked up the candidate's real GitHub repositories earlier in this conversation
    and found ones relevant to this job, pass a short summary of them (name + why relevant) as
    relevant_projects so the letter can reference real work. Leave it empty otherwise — never guess
    or invent project details here.
    """
    user_id = config["configurable"]["user_id"]
    result = build_cover_letter(
        user_id=user_id,
        job_description=job_description,
        resume_summary=session.get("resume_summary") or "",
        contact_info=session.get("extracted_info") or "",
        relevant_projects=relevant_projects,
    )

    remaining_issues = result["remaining_issues"]
    note = (
        f" (validation could not fully confirm this after several attempts: {'; '.join(remaining_issues)} — please review before sending)"
        if remaining_issues
        else ""
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Cover letter drafted and saved.{note}\n\n{result['letter_text']}",
                    tool_call_id=tool_call_id,
                )
            ],
            "last_artifact_path": result["filename"],
            "last_artifact_label": result["label"],
        }
    )


@tool
def generate_tailored_resume(
    job_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    relevant_projects: str = "",
) -> Command:
    """Tailor the user's LaTeX resume template to the given job description and compile it to a downloadable PDF.

    If you already looked up the candidate's real GitHub repositories earlier in this conversation
    and found ones especially relevant to this job, pass them as relevant_projects — include each
    project's real URL. A project with a URL included may be added as a new Projects-section entry
    if it isn't already in the resume; without a URL it can only inform reordering of existing
    content. The tool's own response tells you exactly what happened — rely on that, not on what
    you expect, when describing the result to the user.
    """
    user_id = config["configurable"]["user_id"]
    result = build_tailored_resume(
        user_id=user_id,
        job_description=job_description,
        relevant_projects=relevant_projects,
    )

    if result.get("error"):
        return Command(
            update={
                "messages": [
                    ToolMessage(content=result["error"], tool_call_id=tool_call_id)
                ]
            }
        )

    return Command(
        update={
            "messages": [
                ToolMessage(content=result["message"], tool_call_id=tool_call_id)
            ],
            "last_artifact_path": result["filename"],
            "last_artifact_label": result["label"],
        }
    )


LOCAL_TOOLS = [
    fetch_job_posting_text,
    get_resume_context,
    answer_application_question,
    generate_cover_letter,
    generate_tailored_resume,
]

# Everything except the two document generators. The planned graph's question-answering
# node uses these: it needs real tool use to answer things ("what does this posting say
# about sponsorship?"), but must not be able to produce a resume or cover letter, since
# that path exists to enforce the plan/draft approval gates.
QA_TOOLS = [
    fetch_job_posting_text,
    get_resume_context,
    answer_application_question,
]
