import re

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.resume_bot.shared import BANNED_PHRASES

PLACEHOLDER_PATTERN = re.compile(r"\[[A-Za-z][A-Za-z '.,]{2,40}\]")
BANNED_OPENER = "i am writing to express my interest"

SECTION_PATTERN = re.compile(r"\\section\{([^}]*)\}")
ITEM_PATTERN = re.compile(r"\\item\b")
HREF_PATTERN = re.compile(r"\\href\{([^}]*)\}")

JUDGE_PROMPT_TEMPLATE = """
You are a strict reviewer checking a generated {document_type} against these rules:
{rules}

Document:
{document_text}

Respond with exactly "PASS" if it fully complies with every rule.
Otherwise respond with "FAIL: " followed by one short, specific sentence describing the single
most important violation to fix. Do not comment on anything not covered by the rules above.
"""

COVER_LETTER_JUDGE_RULES = f"""
- Never uses any of these words/phrases or close variants: {", ".join(BANNED_PHRASES)}.
- Never uses bracketed placeholder text (e.g. "[Company Address]", "[Your Name]", "[Today's Date]").
- Never opens with "I am writing to express my interest" or a close variant.
- Never uses an em dash.
- Every specific detail (address, hiring manager name, project mentioned) is either grounded in
  the resume/contact info/job description given, or omitted — nothing is guessed.
"""

RESUME_JUDGE_RULES = """
- Does not invent or fabricate any experience, employer, title, date, skill, or metric that isn't
  already present in the original resume.
- Only rewords existing bullet points/summary/skills to match the job — never adds new claims.
- Every original bullet point and skill still appears somewhere (reordering is fine, dropping is not).
"""


def validate_cover_letter_text(text):
    issues = []
    lower = text.lower()

    for phrase in BANNED_PHRASES:
        if phrase.lower() in lower:
            issues.append(f'uses banned phrase "{phrase}"')

    placeholders = PLACEHOLDER_PATTERN.findall(text)
    if placeholders:
        issues.append(f"contains bracketed placeholder text: {', '.join(placeholders[:3])}")

    if BANNED_OPENER in lower[:250]:
        issues.append('opens with the banned phrase "I am writing to express my interest"')

    if "\u2014" in text:
        issues.append("uses an em dash")

    return issues


def validate_tailored_resume_structure(original_tex, tailored_tex):
    issues = []

    orig_sections = SECTION_PATTERN.findall(original_tex)
    new_sections = SECTION_PATTERN.findall(tailored_tex)
    if orig_sections != new_sections:
        issues.append(
            f"section list changed from {orig_sections} to {new_sections}"
        )

    orig_item_count = len(ITEM_PATTERN.findall(original_tex))
    new_item_count = len(ITEM_PATTERN.findall(tailored_tex))
    if new_item_count < orig_item_count:
        issues.append(
            f"has fewer bullet items than the original ({new_item_count} vs {orig_item_count})"
        )

    orig_hrefs = set(HREF_PATTERN.findall(original_tex))
    new_hrefs = set(HREF_PATTERN.findall(tailored_tex))
    extra_hrefs = new_hrefs - orig_hrefs
    if extra_hrefs:
        issues.append(f"contains links not present in the original resume: {extra_hrefs}")

    return issues


def judge_document(document_type, rules, document_text):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["document_type", "rules", "document_text"],
        template=JUDGE_PROMPT_TEMPLATE,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    verdict = chain.run(
        document_type=document_type, rules=rules, document_text=document_text
    ).strip()
    if verdict.upper().startswith("PASS"):
        return True, ""
    return False, verdict


def generate_with_validation(
    attempt_fn, deterministic_check_fn, judge_rules, document_type, max_attempts=2
):
    """Runs attempt_fn(feedback) up to max_attempts times.

    Each draft is checked with deterministic_check_fn(text) -> list[str] (cheap, no LLM cost).
    Only if that passes does it go through an LLM-judge pass. On any failure, the issues are
    joined into feedback and fed into the next attempt. Returns (final_text, remaining_issues) —
    remaining_issues is empty only if some attempt fully passed both checks.
    """
    feedback = "None."
    last_text = ""
    last_issues = []

    for _ in range(max_attempts):
        last_text = attempt_fn(feedback)
        issues = deterministic_check_fn(last_text)

        if not issues:
            passed, judge_feedback = judge_document(document_type, judge_rules, last_text)
            if passed:
                return last_text, []
            issues = [judge_feedback]

        last_issues = issues
        feedback = "; ".join(issues)

    return last_text, last_issues
