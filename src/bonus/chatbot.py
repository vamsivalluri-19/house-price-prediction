from __future__ import annotations


class RealEstateChatbot:
    """Simple intent-based Q&A assistant for real-estate workflows."""

    def answer(self, question: str) -> str:
        q = question.strip().lower()
        if "best location" in q:
            return "Compare neighborhoods using location popularity index and livability signals."
        if "good price" in q or "list price" in q:
            return "Use predicted price plus confidence-aware recommendation range for listing strategy."
        if "documents" in q:
            return "Validate title, tax records, occupancy certificates, and local compliance checks."
        return "Ask about pricing, location analysis, or listing strategy for tailored guidance."
