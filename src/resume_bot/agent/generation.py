"""Core document-generation logic, independent of how it gets invoked.

Both the ReAct agent's @tool wrappers (tools.py) and the planned graph's nodes
(planned_graph.py) call these functions, so the two entry points can never drift
apart the way they would if each had its own copy of the draft/validate/compile flow.
"""

import datetime
import os
import re
import subprocess

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.resume_bot.shared import (
    COVER_LETTER_PROMPT_TEMPLATE,
    RESUME_TAILORING_PROMPT_TEMPLATE,
    generate_pdf,
    get_job_context,
    sanitize_filename_part,
    strip_code_fence,
)
from src.resume_bot.agent.validation import (
    COVER_LETTER_JUDGE_RULES,
    RESUME_JUDGE_RULES,
    generate_with_validation,
    reconcile_near_duplicate_hrefs,
    surgical_fix,
    validate_cover_letter_text,
    validate_tailored_resume_structure,
)

# Excludes trailing punctuation/closing brackets so "(https://.../repo): description"
# doesn't capture the URL with a stray "):" stuck to the end.
URL_PATTERN = re.compile(r"https?://[^\s)\]},;:'\"<>]+")
HREF_PATTERN = re.compile(r"\\href\{([^}]*)\}")


def artifact_dir(user_id):
    path = os.path.join("data", "agent_artifacts", user_id)
    os.makedirs(path, exist_ok=True)
    return path


def resume_template_path(user_id):
    return f"static/resumes/{user_id}_resume_template.tex"


def build_cover_letter(
    user_id,
    job_description,
    resume_summary="",
    contact_info="",
    relevant_projects="",
    revision_feedback="None.",
):
    """Draft, validate, and save a cover letter PDF.

    Returns {filename, label, letter_text, remaining_issues, company}.
    """
    job_summary, company = get_job_context(job_description)

    llm = ChatOpenAI(temperature=0.5)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = PromptTemplate(
        input_variables=[
            "resume_summary",
            "job_description",
            "contact_info",
            "today",
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
        initial_feedback=revision_feedback or "None.",
    )

    out_dir = artifact_dir(user_id)
    filename = f"CoverLetter{sanitize_filename_part(company)}.pdf"
    with open(os.path.join(out_dir, filename), "wb") as f:
        f.write(generate_pdf(letter_text))

    label = f"Cover Letter for {company}" if company else "Cover Letter"
    return {
        "filename": filename,
        "label": f"{label} (PDF)",
        "letter_text": letter_text,
        "remaining_issues": remaining_issues,
        "company": company,
    }


def build_tailored_resume(
    user_id,
    job_description,
    relevant_projects="",
    revision_feedback="None.",
    debug_trace_path=None,
):
    """Tailor the user's LaTeX resume, validate it, and compile it to PDF.

    Returns {filename, label, message, remaining_issues, company, added_hrefs, error}.
    On a compile/validation failure the closest .tex draft is saved instead of a PDF.
    """
    job_summary, company = get_job_context(job_description)

    user_resume_tex_path = resume_template_path(user_id)
    if not os.path.exists(user_resume_tex_path):
        return {
            "error": (
                f"No resume template found at {user_resume_tex_path}. "
                "Ask the user to place their resume's .tex file there first."
            )
        }

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

    out_dir = artifact_dir(user_id)
    tailored_tex_path = os.path.join(out_dir, "tailored_resume.tex")
    tailored_pdf_path = os.path.join(out_dir, "tailored_resume.pdf")
    for stale_path in (tailored_tex_path, tailored_pdf_path):
        if os.path.exists(stale_path):
            os.remove(stale_path)

    # URLs the caller explicitly vouched for (from relevant_projects) are allowed to appear as
    # new links in the tailored resume — anything else new is treated as a likely fabrication.
    allowed_extra_hrefs = set(URL_PATTERN.findall(relevant_projects or ""))

    def attempt(feedback):
        raw = resume_chain.run(
            tex_source=tex_source,
            job_description=job_summary,
            relevant_projects=relevant_projects or "None found.",
            revision_feedback=feedback,
        )
        candidate = strip_code_fence(raw)
        return reconcile_near_duplicate_hrefs(tex_source, candidate)

    def check(candidate_tex):
        issues = validate_tailored_resume_structure(
            tex_source, candidate_tex, allowed_extra_hrefs=allowed_extra_hrefs
        )
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
        augmented_issues = list(issues)
        if any("failed to compile" in issue.lower() for issue in issues):
            augmented_issues.append(
                "Check every \\href and any other new text for unescaped LaTeX special "
                "characters (especially underscores in GitHub repo names/URLs) and escape them "
                "as \\_, \\&, \\%, \\# — this is the most common cause of a compile failure here."
            )
        fixed = strip_code_fence(surgical_fix("tailored LaTeX resume", text, augmented_issues))
        return reconcile_near_duplicate_hrefs(tex_source, fixed)

    judge_rules = (
        RESUME_JUDGE_RULES
        + "\n\nOriginal Resume (for reference only — every entry already present here is "
        "pre-existing and legitimate, NOT a fabrication, even if it's not mentioned below):\n"
        + tex_source
    )
    if relevant_projects:
        judge_rules += (
            f"\n\nThe following GitHub projects were verified as real and may legitimately appear "
            f"as new Projects-section entries — do not flag these as fabricated:\n{relevant_projects}"
        )

    debug_trace = []
    tailored_tex, remaining_issues = generate_with_validation(
        attempt_fn=attempt,
        deterministic_check_fn=check,
        judge_rules=judge_rules,
        document_type="tailored LaTeX resume",
        fix_fn=fix,
        trace=debug_trace,
        initial_feedback=revision_feedback or "None.",
    )
    trace_path = debug_trace_path or os.environ.get("RESUME_DEBUG_TRACE_PATH")
    if trace_path:
        import json as _json

        with open(trace_path, "w", encoding="utf-8") as f:
            _json.dump(debug_trace, f, indent=2)

    company_sanitized = sanitize_filename_part(company)
    base_filename = f"Resume{company_sanitized}" if company_sanitized else "Resume"

    added_hrefs = []
    if not remaining_issues and os.path.exists(tailored_pdf_path):
        final_name = f"{base_filename}.pdf"
        os.replace(tailored_pdf_path, os.path.join(out_dir, final_name))
        orig_hrefs = set(HREF_PATTERN.findall(tex_source))
        added_hrefs = sorted(set(HREF_PATTERN.findall(tailored_tex)) - orig_hrefs)
        if added_hrefs:
            message = (
                "Tailored resume validated and compiled to PDF, ready for download. Added new "
                f"Projects-section entries for: {', '.join(added_hrefs)}."
            )
        else:
            message = (
                "Tailored resume validated and compiled to PDF, ready for download. No new "
                "projects were added — existing content was reworded/reordered only."
            )
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

    return {
        "filename": final_name,
        "label": f"Tailored Resume ({final_name.rsplit('.', 1)[-1].upper()})",
        "message": message,
        "remaining_issues": remaining_issues,
        "company": company,
        "added_hrefs": added_hrefs,
        "error": None,
    }
