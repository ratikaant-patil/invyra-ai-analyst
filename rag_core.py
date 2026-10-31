"""Invyra answering logic: retrieve from business files, answer, and clean the model output."""
import re

from langchain_core.runnables import RunnableLambda

# The model is told to reply with exactly this when the files don't contain the answer.
# That is a far more reliable signal than guessing from phrases like "I don't know".
NOT_FOUND = "NOT_FOUND_IN_DOCUMENTS"

_llm = None


def set_llm_model(llm):
    global _llm
    _llm = llm


def _no_think_suffix():
    """qwen3 models obey '/no_think' in the prompt. Other models don't need it."""
    return "\n/no_think" if "qwen3" in str(getattr(_llm, "model", "")).lower() else ""


def clean_output(text) -> str:
    """Removes the model's reasoning so only the final answer is shown.

    Handles <think>...</think>, a lone </think> (opening tag missing), an unfinished
    <think> block, and the 'Thinking... ...done thinking.' format.
    """
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*Thinking\.\.\..*?\.\.\.done thinking\.\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


RAG_TEMPLATE = """You are Invyra, an expert business analytics assistant.

Answer the question using ONLY the information below. It comes from the user's own business files.
- Be clear and concise. Use short paragraphs or bullet points when it helps.
- Include specific figures, names and dates when they are relevant.
- If the information below does not answer the question, reply with exactly: {not_found}
- Never show your reasoning. Never mention "the context" or "the information provided".

Information:
{context}

Question: {question}{suffix}"""

GENERAL_TEMPLATE = """You are Invyra, an expert business analytics assistant.

The user's business files do not contain the answer to this question, so answer from general
business knowledge. Be clear and concise, and say so if you are not sure.
Never show your reasoning.

Question: {question}{suffix}"""


def _format_context(docs):
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "")
        page = d.metadata.get("page")
        row = d.metadata.get("row")
        label = f"{source}, page {page}" if page else (f"{source}, row {row}" if row else source)
        blocks.append(f"[{label}]\n{d.page_content}")
    return "\n\n".join(blocks)


def _source_labels(docs):
    labels = []
    for d in docs:
        source = d.metadata.get("source", "")
        page = d.metadata.get("page")
        row = d.metadata.get("row")
        label = f"{source}, p. {page}" if page else source
        if row and not page:
            label = source  # rows are too granular to list; show the file
        if label and label not in labels:
            labels.append(label)
    return labels[:6]


def answer_from_documents(question, retriever):
    """Returns {"answer": str, "found": bool, "sources": [str]}."""
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "", "found": False, "sources": []}

    prompt = RAG_TEMPLATE.format(
        not_found=NOT_FOUND,
        context=_format_context(docs),
        question=question,
        suffix=_no_think_suffix(),
    )
    answer = clean_output(_llm.invoke(prompt))
    found = bool(answer) and NOT_FOUND not in answer
    return {"answer": answer if found else "", "found": found, "sources": _source_labels(docs) if found else []}


def answer_with_local_model(question):
    """Plain answer from the local model, used when the documents have nothing."""
    prompt = GENERAL_TEMPLATE.format(question=question, suffix=_no_think_suffix())
    return clean_output(_llm.invoke(prompt))


def get_rag_chain(retriever):
    """Kept for compatibility: a runnable that takes a question and returns the answer text."""
    return RunnableLambda(lambda q: answer_from_documents(q, retriever)["answer"])