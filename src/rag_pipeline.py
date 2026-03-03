from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


PERSIST_DIRECTORY = "vectorstore"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "phi3:mini"

@dataclass
class RAGConfig:
    top_k: int = 3
    min_relevance: float = 0.6


SYSTEM_PROMPT = """You are an AI assistant for a large automotive company.

Answer the user's question only using the provided context.
Ignore any context that is not directly relevant to the user's question.
If the answer is not contained in the context, respond with:
"Sorry, I cannot help with this question based on the provided documents."

Do not add a 'Sources:' section inside the answer text.
Sources will be handled separately by the application.

Keep the answer concise, factual, and professional.
"""


def _load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)


def retrieve(query: str, config: RAGConfig) -> List:
    vectorstore = _load_vectorstore()
    docs_with_relevance = vectorstore.similarity_search_with_relevance_scores(
        query, k=config.top_k
    )
    filtered = [doc for doc, score in docs_with_relevance if score >= config.min_relevance]

    if not filtered and docs_with_relevance:
        filtered = [docs_with_relevance[0][0]]

    return filtered


def _format_context(docs: List) -> Tuple[str, List[str]]:
    context_blocks = []
    sources = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        filename = src.split("\\")[-1].split("/")[-1]
        title = filename.replace("doc_", "").replace(".txt", "").replace("_", " ").strip()
        sources.append(title)
        context_blocks.append(d.page_content)

    seen = set()
    unique_sources = []
    for s in sources:
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)

    return "\n\n---\n\n".join(context_blocks), unique_sources


def generate(query: str, docs: List) -> tuple[str, List[str]]:
    context, sources = _format_context(docs)

    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    llm = OllamaLLM(model=CHAT_MODEL)
    answer = llm.invoke(prompt).strip()

    fallback = "Sorry, I cannot help with this question based on the provided documents."
    if fallback in answer:
        return answer, []

    return answer, sources


def ask(query: str, config: RAGConfig | None = None) -> tuple[str, List[str]]:
    if config is None:
        config = RAGConfig()

    docs = retrieve(query, config)
    return generate(query, docs)

    #TEST

if __name__ == "__main__":
    answer, sources = ask("What is covered by the warranty?")
    print("Answer:\n", answer)
    print("\nSources:")
    for s in sources:
        print("-", s)