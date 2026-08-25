import difflib
import re

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.resume_bot.shared import BANNED_PHRASES

PLACEHOLDER_PATTERN = re.compile(r"\[[A-Za-z][A-Za-z '.,]{2,40}\]")
BANNED_OPENER_PATTERN = re.compile(
    r"\b(writing|excited|eager|pleased|thrilled|delighted)\s+to\s+"
    r"(express my interest|apply for|join|be considered for|submit my application)\b"
)

SECTION_PATTERN = re.compile(r"\\section\{([^}]*)\}")
ITEM_PATTERN = re.compile(r"\\item\b")
HREF_PATTERN = re.compile(r"\\href\{([^}]*)\}")


def _normalize_url(url):
    return re.sub(r"[^a-z0-9]", "", url.lower())


def reconcile_near_duplicate_hrefs(original_tex, candidate_tex):
    """Models have a strong habit of "fixing" what they perceive as a typo in an existing URL
    (e.g. rewriting pyspark-recommentation to pyspark-recommendation) even when explicitly told
    not to touch existing links. Rather than keep fighting that with more prompt instructions,
    silently correct any such near-duplicate back to the original exact URL before validation."""
    orig_hrefs = HREF_PATTERN.findall(original_tex)
    candidate_hrefs = set(HREF_PATTERN.findall(candidate_tex))
    orig_set = set(orig_hrefs)

    fixed_tex = candidate_tex
    for href in candidate_hrefs - orig_set:
        normalized = _normalize_url(href)
        best_match, best_ratio = None, 0.0
        for o in orig_hrefs:
            if o == href:
                continue
            ratio = difflib.SequenceMatcher(None, normalized, _normalize_url(o)).ratio()
            if ratio > best_ratio:
                best_match, best_ratio = o, ratio
        match = best_match if best_ratio >= 0.85 else None
        if match:
            fixed_tex = fixed_tex.replace(href, match)

    return fixed_tex

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
  already present in the original resume or in the verified GitHub projects it was given.
- Every original bullet point and skill still appears somewhere (reordering is fine, dropping is not).
- Any new Projects-section entry is only acceptable if it corresponds to one of the specific,
  verified GitHub repositories it was given (real name/description/URL) — never a project it made up.
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

    if BANNED_OPENER_PATTERN.search(lower[:250]):
        issues.append('opens with a "___ to express my interest" cliché phrase')

    if "\u2014" in text:
        issues.append("uses an em dash")

    return issues


def validate_tailored_resume_structure(original_tex, tailored_tex, allowed_extra_hrefs=None):
    """Checks structural preservation. allowed_extra_hrefs (a set of URLs) lets new links through
    when they correspond to real, verified GitHub projects the caller explicitly vouched for
    (e.g. extracted from the relevant_projects text) — anything else new is flagged as a likely
    fabrication, since the tailoring prompt is only supposed to reword/reorder existing content."""
    issues = []
    allowed_extra_hrefs = allowed_extra_hrefs or set()

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
    unexplained_hrefs = new_hrefs - orig_hrefs - allowed_extra_hrefs
    if unexplained_hrefs:
        issues.append(
            f"contains links not present in the original resume and not in the vouched-for "
            f"GitHub projects: {unexplained_hrefs}"
        )

    return issues


SURGICAL_FIX_PROMPT_TEMPLATE = """
The following {document_type} has these specific problems: {issues}

Make the smallest possible edit to fix exactly these problems — rephrase only the flagged
words, phrases, or lines. Do not rewrite, reorder, or otherwise change anything else about the
document. Return the complete corrected document and nothing else: no explanation, no
commentary, no markdown code fences.

Document:
{document_text}
"""


def surgical_fix(document_type, document_text, issues):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        input_variables=["document_type", "issues", "document_text"],
        template=SURGICAL_FIX_PROMPT_TEMPLATE,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(
        document_type=document_type,
        issues="; ".join(issues),
        document_text=document_text,
    ).strip()


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
    attempt_fn,
    deterministic_check_fn,
    judge_rules,
    document_type,
    max_attempts=2,
    fix_fn=None,
    trace=None,
):
    """Runs attempt_fn(feedback) up to max_attempts times.

    Each draft is checked with deterministic_check_fn(text) -> list[str] (cheap, no LLM cost).
    Only if that passes does it go through an LLM-judge pass. On any failure, the issues are
    joined into feedback and fed into the next full-regeneration attempt.

    If issues remain after max_attempts, a full fresh regeneration can just as easily reintroduce
    the same (or a different) cliché by chance, so as a last resort this makes one targeted
    surgical-fix pass on the last draft — asking the model to fix only the flagged issues rather
    than rewrite from scratch — before giving up.

    Returns (final_text, remaining_issues); remaining_issues is empty only if some draft fully
    passed both checks (possibly after the surgical fix).
    """
    feedback = "None."
    last_text = ""
    last_issues = []

    for attempt_num in range(1, max_attempts + 1):
        last_text = attempt_fn(feedback)
        issues = deterministic_check_fn(last_text)
        judge_verdict = None

        if not issues:
            passed, judge_feedback = judge_document(document_type, judge_rules, last_text)
            judge_verdict = "PASS" if passed else judge_feedback
            if passed:
                if trace is not None:
                    trace.append(
                        {
                            "stage": f"attempt-{attempt_num}",
                            "text": last_text,
                            "deterministic_issues": [],
                            "judge_verdict": judge_verdict,
                            "result": "PASSED",
                        }
                    )
                return last_text, []
            issues = [judge_feedback]

        if trace is not None:
            trace.append(
                {
                    "stage": f"attempt-{attempt_num}",
                    "text": last_text,
                    "deterministic_issues": issues if judge_verdict is None else [],
                    "judge_verdict": judge_verdict,
                    "result": "FAILED",
                }
            )

        last_issues = issues
        feedback = "; ".join(issues)

    if not last_issues:
        return last_text, last_issues

    fixer = fix_fn or (lambda text, issues: surgical_fix(document_type, text, issues))
    fixed_text = fixer(last_text, last_issues)
    deterministic_fixed_issues = deterministic_check_fn(fixed_text)
    fixed_issues = deterministic_fixed_issues
    fix_judge_verdict = None
    if not fixed_issues:
        passed, judge_feedback = judge_document(document_type, judge_rules, fixed_text)
        fix_judge_verdict = "PASS" if passed else judge_feedback
        if not passed:
            fixed_issues = [judge_feedback]

    if trace is not None:
        trace.append(
            {
                "stage": "surgical-fix",
                "text": fixed_text,
                "deterministic_issues": deterministic_fixed_issues,
                "judge_verdict": fix_judge_verdict,
                "result": "PASSED" if not fixed_issues else "FAILED",
                "issues_given_to_fixer": last_issues,
            }
        )

    # A compile failure is never an acceptable trade for some other issue, even a "smaller" one by
    # count — e.g. a surgical fix that resolves a fabrication complaint but leaves behind an
    # unescaped underscore in a new \href introduces a NEW, fatal problem that raw issue-count
    # comparison would otherwise miss (1 issue -> 1 different issue looks like "no worse").
    if _has_compile_failure(fixed_issues) and not _has_compile_failure(last_issues):
        return last_text, last_issues

    # Otherwise, prefer the fixed draft unless it's strictly worse — a fix that doesn't fully
    # satisfy the judge is still closer to correct than reverting to the pre-fix draft.
    if len(fixed_issues) <= len(last_issues):
        return fixed_text, fixed_issues

    return last_text, last_issues


def _has_compile_failure(issues):
    return any("failed to compile" in issue.lower() for issue in issues)
