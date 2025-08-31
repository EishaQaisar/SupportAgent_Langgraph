# src/agent/nodes/draft_generator.py
from typing import List

def generate_draft(subject: str, description: str, category: str, docs: List[str]) -> str:
    """
    Generate a polite, professional customer support draft using the provided context.
    No LLM is used — the response is composed with code.
    """

    if not docs:
        return (
            f"Thank you for reaching out. I see your request is related to {category.lower()}. "
            f"Unfortunately, I couldn’t find specific details in our resources about '{subject}'. "
            "Please check our Help Center or provide more information so we can assist you further. "
            "We truly appreciate your patience."
        )

    # Build bullet list from context
    context_points = "\n".join([f"- {doc}" for doc in docs])

    # Construct polite response
    draft = (
        f"Thank you for reaching out. I understand your request is related to {category.lower()} category.\n\n"
        f"Here’s some information that may help you:\n"
        f"{context_points}\n\n"
        "I hope this information is useful. If you have any other questions, feel free to let us know — "
        "we’re always happy to assist."
    )

    return draft
