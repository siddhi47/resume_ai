"""Chat model factory.

Lets the whole app run against either the OpenAI API or a locally hosted Ollama
model, chosen at runtime with LLM_PROVIDER. The exported name is deliberately
``ChatOpenAI`` so the ~17 existing call sites stay untouched - only their import
line changes.

    LLM_PROVIDER=openai   (default)  use the OpenAI API as before
    LLM_PROVIDER=ollama              use OLLAMA_MODEL via OLLAMA_BASE_URL

Under Ollama the per-call ``model=`` argument (gpt-4o, gpt-4o-mini, o4-mini...)
is ignored, since those names mean nothing locally; every call resolves to
OLLAMA_MODEL instead.
"""

import os

# Arguments that only make sense for the OpenAI backend.
_OPENAI_ONLY = (
    "model",
    "model_name",
    "openai_api_key",
    "openai_api_base",
    "max_completion_tokens",
)


def _use_ollama() -> bool:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower() == "ollama"


def ChatOpenAI(**kwargs):  # noqa: N802 - name kept to avoid touching call sites
    """Return a chat model for the configured provider."""
    if not _use_ollama():
        from langchain_openai import ChatOpenAI as _ChatOpenAI

        return _ChatOpenAI(**kwargs)

    from langchain_ollama import ChatOllama

    for key in _OPENAI_ONLY:
        kwargs.pop(key, None)

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        # qwen3.5 exposes a "thinking" mode that will otherwise spend thousands
        # of tokens reasoning before answering, long enough to stall a request.
        reasoning=False,
        num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "2048")),
        **kwargs,
    )


def needs_explicit_plan() -> bool:
    """Whether the backend needs edits spelled out before it will make them.

    Asked open-endedly to "tailor this resume", a small local model reproduces its
    input almost verbatim - the output compiles perfectly and changes nothing, which
    is worse than failing. The same model applies edits reliably when told exactly
    what to change, so for local backends we plan the edits in a separate cheap pass
    first. Hosted frontier models do not need the scaffolding.
    """
    return _use_ollama()


def describe() -> str:
    """One-line description of the active backend, for logging."""
    if _use_ollama():
        return (
            f"ollama:{os.getenv('OLLAMA_MODEL', 'qwen3.5:4b')} "
            f"@ {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}"
        )
    return "openai"
