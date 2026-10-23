# Grounded Enterprise Knowledge Assistant

A local-first portfolio project for learning enterprise grounding and retrieval concepts.

## What it does

- Loads Markdown documents from `data/documents/`.
- Splits documents into chunks while retaining source and section metadata.
- Uses TF-IDF retrieval to find relevant evidence.
- Abstains when the best retrieval score is below a configurable threshold.
- Returns answers as cited evidence excerpts rather than inventing a generated answer.
- Exposes a FastAPI endpoint and includes automated tests.

## Requirements

- Python 3.11 or newer
- macOS, Windows, or Linux

## Run locally on macOS

```bash
cd aws-fde-ground-assistant
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

## Example request

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long are application logs retained?"}'
```

Example response shape:

```json
{
  "question": "How long are application logs retained?",
  "answered": true,
  "answer": "Application logs are retained for 30 days unless a legal hold requires longer retention.",
  "citations": [
    {
      "source": "data-retention.md",
      "section": "Application logs",
      "excerpt": "Application logs are retained for 30 days unless a legal hold requires longer retention.",
      "score": 0.5
    }
  ],
  "reason": null
}
```

Scores vary by scikit-learn version and query. Treat the score as a ranking signal, not a calibrated confidence probability.

## Important limitation

This is a learning prototype, not a production-ready security or compliance system. TF-IDF is lexical retrieval; it may miss paraphrases and semantic matches. The returned answer is an evidence excerpt, not an LLM-generated synthesis. The documents are fictional examples.

## Suggested next milestones

1. Add chunking/retrieval evaluation with a fixed question set.
2. Add semantic embeddings and compare recall against TF-IDF.
3. Add a local or hosted LLM only after retrieval quality is measured.
4. Add document access-control metadata and tests before using sensitive documents.
5. Add structured logging, latency metrics, and deployment documentation.
