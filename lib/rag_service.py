from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> set[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {
        token.strip("'")
        for token in raw_tokens
        if len(token.strip("'")) > 1 and token.strip("'") not in STOPWORDS
    }


def document_search_text(document: dict[str, Any]) -> str:
    tags = " ".join(document.get("tags", []))
    return (
        f"{document.get('title', '')} "
        f"{document.get('category', '')} "
        f"{tags} "
        f"{document.get('text', '')}"
    )


def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    """Score a document using keyword overlap"""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document_search_text(document))
    title_tokens = tokenize(document.get("title", ""))
    matched_terms = query_tokens.intersection(document_tokens)
    title_matches = query_tokens.intersection(title_tokens)
    score = len(matched_terms) + (0.5 * len(title_matches))
    return {
        "document": document,
        "score": score,
        "matched_terms": sorted(matched_terms),
    }


def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    
    scored_matches = [score_document(query, document) for document in documents]
    relevant_matches = [
        match for match in scored_matches if match["score"] >= minimum_score
    ]
    return sorted(relevant_matches, key=lambda match: match["score"], reverse=True)[:limit]



def format_context(context_matches: list[dict[str, Any]]) -> str:
    if not context_matches:
        return "No relevant context was found in the approved company documents."

    context_blocks = []
    for match in context_matches:
        document = match["document"]
        context_blocks.append(
            "\n".join(
                [
                    f"Source ID: {document['id']}",
                    f"Title: {document['title']}",
                    f"Category: {document['category']}",
                    f"Content: {document['text']}",
                ]
            )
        )
    return "\n\n---\n\n".join(context_blocks)


def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    context_block = format_context(context_matches)
    return f"""You are an internal company documentation assistant.
Instructions:
Use only the provided context to answer the user's question.
If the context does not contain enough information, say that the approved company documents do not contain enough information.
Do not invent policies, URLs, phone numbers, timelines, approvals, or escalation paths.

Context:
{context_block}

Question:
{query.strip()}

Response requirements:
- Answer in 2-4 concise sentences.
- Use a helpful workplace-support tone.
- Mention the source ID or source IDs used.
"""


def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    document = match["document"]
    return {
        "id": document["id"],
        "title": document["title"],
    }
