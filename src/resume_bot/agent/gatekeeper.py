import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

URL_PATTERN = re.compile(r"https?://\S+")

REJECTION_MESSAGE = (
    "I can only help with job applications — resumes, cover letters, job postings, and related "
    "questions. Try asking about one of those instead."
)

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a scope gate for a job-application assistant chatbot. Your only job is to catch
messages that are CLEARLY, UNAMBIGUOUSLY unrelated to jobs or careers — e.g. general programming/
coding help unrelated to the user's own resume, trivia, creative writing requests, or any other
task with nothing to do with a job search. Reject only these obvious cases.

Default to IN_SCOPE for everything else, including:
- Greetings, thanks, and questions about what the assistant can do.
- A pasted block of text that looks like a job description, job title, company info, or resume
  content — even with no question or request attached. Users often just paste this material
  directly; it does not need to be phrased as a request to be in scope.
- Any request, however phrased, involving the user's own resume, cover letter, job application,
  or career.

If you are not CONFIDENT the message is clearly unrelated to jobs/careers, answer IN_SCOPE.

Message:
{message}

Answer with exactly one word: IN_SCOPE or OUT_OF_SCOPE."""
)

_router_llm = None


def _get_router_llm():
    # Lazy, same reason as graph.py's _get_llm(): avoid building a client before
    # app.py's load_dotenv() has run.
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _router_llm


async def is_in_scope(message: str) -> bool:
    # A pasted URL is the core UX for the job-posting-fetch feature, and the LLM classifier
    # proved unreliable on bare/lightly-labeled links (rejected even "This is a job posting
    # URL: <link>"). Treat any URL as in-scope deterministically rather than relying on the
    # classifier for this case — cheaper too, since it skips the LLM call entirely.
    if URL_PATTERN.search(message):
        return True

    chain = _ROUTER_PROMPT | _get_router_llm()
    result = await chain.ainvoke({"message": message})
    return result.content.strip().upper().startswith("IN_SCOPE")
