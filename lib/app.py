from __future__ import annotations

from flask import Flask, jsonify, request

from lib.ai_client import generate_response
from lib.company_documents import COMPANY_DOCUMENTS
from lib.rag_service import build_prompt, retrieve_context, source_metadata


def create_app():
    app = Flask(__name__)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask_question():
        payload = request.get_json(silent=True) or {}
        query = payload.get("query")

        if not isinstance(query, str) or not query.strip():
            return (
                jsonify(
                    {
                        "error": "A non-empty 'query' string is required.",
                        "example": {"query": "How do I request software access?"},
                    }
                ),
                400,
            )

        clean_query = query.strip()
        context_matches = retrieve_context(clean_query, COMPANY_DOCUMENTS, limit=2)
        sources = [source_metadata(match) for match in context_matches]

        if not context_matches:
            return (
                jsonify(
                    {
                        "query": clean_query,
                        "answer": "The approved company documents do not contain enough information to answer that question.",
                        "sources": [],
                    }
                ),
                200,
            )
        prompt = build_prompt(clean_query, context_matches)

        try:
            answer = generate_response(prompt)
        except RuntimeError as error:
            return (
                jsonify(
                    {
                        "error": f"The model service could not generate a response: {error}",
                    }
                ),
                503,
            )

        return jsonify({"query": clean_query, "answer": answer, "sources": sources}), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
