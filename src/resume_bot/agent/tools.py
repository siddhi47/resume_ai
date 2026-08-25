import os
import subprocess
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
    COVER_LETTER_PROMPT_TEMPLATE,
    RESUME_TAILORING_PROMPT_TEMPLATE,
    GENERIC_QA_PROMPT_TEMPLATE,
    generate_pdf,
    get_job_context,
    sanitize_filename_part,
    strip_code_fence,
)
from src.resume_bot.agent.validation import (
    COVER_LETTER_JUDGE_RULES,
    RESUME_JUDGE_RULES,
    generate_with_validation,
    surgical_fix,
    validate_cover_letter_text,
    validate_tailored_resume_structure,
)

import datetime


def _artifact_dir(user_id: str) -> str:
    path = os.path.join("data", "agent_artifacts", user_id)
    os.makedirs(path, exist_ok=True)
    return path


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
    resume_summary = session.get("resume_summary") or ""
    contact_info = session.get("extracted_info") or ""
    job_summary, company = get_job_context(job_description)

    llm = ChatOpenAI(temperature=0.5)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = PromptTemplate(
        input_variables=[
            "resume_summary",
            "job_description",
            "contact_info",
            "relevant_projects",
            "revision_feedback",
        ],
        template=COVER_LETTER_PROMPT_TEMPLATE,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    def attempt(feedback):
        return chain.run(
            resume_summary=resume_summary,
            job_description=job_summary,
            contact_info=contact_info,
            today=today,
            relevant_projects=relevant_projects or "None found.",
            revision_feedback=feedback,
        )

    letter_text, remaining_issues = generate_with_validation(
        attempt_fn=attempt,
        deterministic_check_fn=validate_cover_letter_text,
        judge_rules=COVER_LETTER_JUDGE_RULES,
        document_type="cover letter",
    )

    out_dir = _artifact_dir(user_id)
    filename = f"CoverLetter{sanitize_filename_part(company)}.pdf"
    with open(os.path.join(out_dir, filename), "wb") as f:
        f.write(generate_pdf(letter_text))

    label = f"Cover Letter for {company}" if company else "Cover Letter"
    note = (
        f" (validation could not fully confirm this after several attempts: {'; '.join(remaining_issues)} — please review before sending)"
        if remaining_issues
        else ""
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Cover letter drafted and saved.{note}\n\n{letter_text}",
                    tool_call_id=tool_call_id,
                )
            ],
            "last_artifact_path": filename,
            "last_artifact_label": f"{label} (PDF)",
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
    and found ones especially relevant to this job, pass a short summary of them (name + why
    relevant) as relevant_projects. This only affects which EXISTING resume bullets/skills get
    emphasized or reordered — the resume can't add new bullets, projects, or links, so don't expect
    GitHub findings to appear as new content, only as a signal for what to prioritize.
    """
    user_id = config["configurable"]["user_id"]
    job_summary, company = get_job_context(job_description)

    user_resume_tex_path = f"static/resumes/{user_id}_resume_template.tex"
    if not os.path.exists(user_resume_tex_path):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"No resume template found at {user_resume_tex_path}. "
                            "Ask the user to place their resume's .tex file there first."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    with open(user_resume_tex_path, "r", encoding="utf-8", errors="replace") as f:
        tex_source = f.read()

    resume_llm = ChatOpenAI(temperature=0, model="gpt-4o")
    resume_prompt = PromptTemplate(
        input_variables=[
            "tex_source",
            "job_description",
            "relevant_projects",
            "revision_feedback",
        ],
        template=RESUME_TAILORING_PROMPT_TEMPLATE,
    )
    resume_chain = LLMChain(llm=resume_llm, prompt=resume_prompt)

    out_dir = _artifact_dir(user_id)
    tailored_tex_path = os.path.join(out_dir, "tailored_resume.tex")
    tailored_pdf_path = os.path.join(out_dir, "tailored_resume.pdf")
    for stale_path in (tailored_tex_path, tailored_pdf_path):
        if os.path.exists(stale_path):
            os.remove(stale_path)

    def attempt(feedback):
        raw = resume_chain.run(
            tex_source=tex_source,
            job_description=job_summary,
            relevant_projects=relevant_projects or "None found.",
            revision_feedback=feedback,
        )
        return strip_code_fence(raw)

    def check(candidate_tex):
        issues = validate_tailored_resume_structure(tex_source, candidate_tex)
        if issues:
            return issues

        with open(tailored_tex_path, "w", encoding="utf-8") as f:
            f.write(candidate_tex)
        compile_result = subprocess.run(
            ["tectonic", "--outdir", out_dir, tailored_tex_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if compile_result.returncode != 0 or not os.path.exists(tailored_pdf_path):
            stderr_lines = (compile_result.stderr or "").strip().splitlines()
            detail = stderr_lines[-1] if stderr_lines else "unknown compile error"
            return [f"LaTeX failed to compile: {detail}"]
        return []

    def fix(text, issues):
        return strip_code_fence(surgical_fix("tailored LaTeX resume", text, issues))

    tailored_tex, remaining_issues = generate_with_validation(
        attempt_fn=attempt,
        deterministic_check_fn=check,
        judge_rules=RESUME_JUDGE_RULES,
        document_type="tailored LaTeX resume",
        fix_fn=fix,
    )

    company_sanitized = sanitize_filename_part(company)
    base_filename = f"Resume{company_sanitized}" if company_sanitized else "Resume"

    if not remaining_issues and os.path.exists(tailored_pdf_path):
        final_name = f"{base_filename}.pdf"
        os.replace(tailored_pdf_path, os.path.join(out_dir, final_name))
        message = "Tailored resume validated and compiled to PDF, ready for download."
    else:
        with open(tailored_tex_path, "w", encoding="utf-8") as f:
            f.write(tailored_tex)
        final_name = f"{base_filename}.tex"
        os.replace(tailored_tex_path, os.path.join(out_dir, final_name))
        issue_text = "; ".join(remaining_issues) if remaining_issues else "unknown issue"
        message = (
            f"The tailored resume didn't pass validation after multiple attempts ({issue_text}), "
            "so I saved the closest .tex draft instead — you can review or compile it manually "
            "(e.g. via Overleaf)."
        )

    return Command(
        update={
            "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
            "last_artifact_path": final_name,
            "last_artifact_label": f"Tailored Resume ({final_name.rsplit('.', 1)[-1].upper()})",
        }
    )


LOCAL_TOOLS = [
    fetch_job_posting_text,
    get_resume_context,
    answer_application_question,
    generate_cover_letter,
    generate_tailored_resume,
]
