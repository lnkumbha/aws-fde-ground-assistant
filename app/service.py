from app.knowledge_base import KnowledgeBase
from app.schemas import AskResponse, Citation


class GroundedAssistant:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        min_score: float = 0.12,
        max_results: int = 3,
    ):
        self.knowledge_base = knowledge_base
        self.min_score = min_score
        self.max_results = max_results

    def ask(self, question: str) -> AskResponse:
        hits = self.knowledge_base.search(question, limit=self.max_results)
        if not hits or hits[0].score < self.min_score:
            return AskResponse(
                question=question,
                answered=False,
                answer=(
                    "I couldn't find enough evidence in the available documents "
                    "to answer that question."
                ),
                citations=[],
                reason="insufficient_evidence",
            )

        # Deliberately return evidence excerpts instead of generating unsupported synthesis.
        selected = [hit for hit in hits if hit.score >= self.min_score]
        citations = [
            Citation(
                source=hit.chunk.source,
                section=hit.chunk.section,
                excerpt=hit.chunk.text,
                score=round(hit.score, 4),
            )
            for hit in selected
        ]
        answer = " ".join(citation.excerpt for citation in citations[:2])
        return AskResponse(
            question=question,
            answered=True,
            answer=answer,
            citations=citations,
        )
