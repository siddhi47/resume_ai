from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

REJECTION_MESSAGE = (
    "I can only help with job applications — resumes, cover letters, job postings, and related "
    "questions. Try asking about one of those instead."
)

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a scope gate for a job-application assistant chatbot. The chatbot's purpose is
helping with: job applications, job postings, resumes, cover letters, the candidate's own GitHub
projects/skills in that context, greetings, and questions about what the chatbot itself can do.

Classify the following user message as exactly one word: IN_SCOPE or OUT_OF_SCOPE.

IN_SCOPE includes: greetings ("hi", "hello"), asking what the assistant can help with or how to
use it, and anything about the user's own job search, resume, cover letters, or job postings.

OUT_OF_SCOPE is for requests unrelated to the user's own job search — e.g. general programming/
coding help unrelated to their resume, trivia, unrelated tasks — even if phrased politely or
disguised as job-related.

When genuinely unsure, prefer IN_SCOPE — only reject messages that are clearly about something
else entirely.

Message:
{message}

Answer with exactly one word."""
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
    chain = _ROUTER_PROMPT | _get_router_llm()
    result = await chain.ainvoke({"message": message})
    return result.content.strip().upper().startswith("IN_SCOPE")
