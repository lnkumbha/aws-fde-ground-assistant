from pathlib import Path

from app.knowledge_base import KnowledgeBase, load_markdown_documents


def test_loads_documents_and_preserves_sections():
    docs = Path(__file__).resolve().parents[1] / "data" / "documents"
    chunks = load_markdown_documents(docs, chunk_words=80, overlap=10)
    assert chunks
    assert any(c.source == "data-retention.md" for c in chunks)
    assert any(c.section == "Application logs" for c in chunks)


def test_retrieval_finds_relevant_retention_evidence():
    docs = Path(__file__).resolve().parents[1] / "data" / "documents"
    kb = KnowledgeBase(load_markdown_documents(docs))
    hits = kb.search("How long are application logs retained?", limit=3)
    assert hits
    assert hits[0].chunk.source == "data-retention.md"
    assert "30 days" in hits[0].chunk.text


def test_empty_knowledge_base_returns_no_hits():
    assert KnowledgeBase([]).search("anything") == []
