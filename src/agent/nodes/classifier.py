# src/agent/nodes/classifier.py

from transformers import pipeline

# Load a tiny BERT model for classification (very small ~15MB)
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


CATEGORIES = ["Billing", "Technical", "Security", "General"]

def classify_ticket(subject: str, description: str) -> str:
    """
    Classify a support ticket into one of the predefined categories.
    Runs fully offline using Hugging Face transformers.
    """

    text = f"""
You are a support ticket classifier. 
Choose the single best category for this ticket from: {", ".join(CATEGORIES)}.

Definitions:
- Billing: Anything about payments, invoices, refunds, subscriptions, or charges
- Technical: Bugs, errors, features not working, installation issues
- Security: Account breaches, password reset, suspicious activity, account hacked
- General: Other inquiries not covered above or like talking to the support team

Ticket:
Subject: {subject}
Description: {description}
"""


    try:
        result = classifier(text, candidate_labels=CATEGORIES)
        print("Classification Scores:", result)

        # pick the label with highest score
        best_label = result["labels"][0]
        return best_label
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return "General"
