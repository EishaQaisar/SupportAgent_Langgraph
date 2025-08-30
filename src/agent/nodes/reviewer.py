# src/agent/nodes/reviewer.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

CATEGORIES = ["Billing", "Technical", "Security", "General"]

def review_response(draft: str, subject: str, desc: str, category: str) -> dict:
    """
    Review the draft response for compliance with support guidelines AND check if category is correct.
    
    Returns:
        {"status": "Approved", "raw": "..."}
        or
        {"status": "Rejected", "feedback": "...", "correct_category": "Billing", "raw": "..."}
    """
    print("Current category is:",category)
    query = f"subject: {subject}, description: {desc}"
    categories_str = ", ".join(CATEGORIES)

    prompt = f"""
You are a strict customer support quality assurance reviewer. 
You must reject the draft if it violates any of these rules:
❌ Reject if:
1. Category Mismatch:
   - The assigned category (“{category}”) for the query (“{query}”) does not match the correct category from:
     - Billing: Payments, invoices, refunds, subscriptions, or charges
     - Technical: Bugs, errors, broken features, installation issues
     - Security: Account breaches, password resets, suspicious activity, hacked accounts
     - General: Any other inquiries or general support requests

2. Policy / Quality Violations:
   - Offers refunds, discounts, or financial commitments
   - Promises something support cannot guarantee (overpromising)
   - Provides sensitive security advice (e.g., password resets, authentication bypass)
   - Is rude, unprofessional, or unclear
   - Is inaccurate, irrelevant, or unhelpful
   - States that it itself will take the action (e.g., “I’ll reset your password for you”)

✅ Approve if:
- It is accurate, helpful, polite, and compliant with the rules above.
- It is categorized correctly.
- It answers the user's query: {query}

Here is the draft response:

--- DRAFT START ---
{draft}
--- DRAFT END ---

The provided category is: {category}

Your task:
1. Decide if the draft is Approved or Rejected.
2. If Rejected because of wrong category, specify the correct category from: [{categories_str}].

Respond with ONLY one of the following formats:
- "Approved"
- "Rejected: <CorrectCategory> <reason> "
"""

    try:
        completion = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "http://localhost",  # optional
                "X-Title": "Support QA Agent",       # optional
            },
        )

        output = completion.choices[0].message.content.strip()
        print(f"[DEBUG] Raw reviewer output:\n{output}\n")

        cleaned = output.strip().lower().replace('"', '').strip()

        if cleaned.startswith("approved"):
            return {"status": "Approved", "raw": output}

        elif cleaned.startswith("rejected"):
            feedback = output
            correct_category = category
              # Extract the first word after "Rejected:"
            try:
                parts = feedback.split()
                if len(parts) > 1:
                    suggested_cat = str(parts[1].strip())
                    # Compare with provided category
                    print("suggested cat:", suggested_cat)
                    print("current", category)
                    if suggested_cat.lower() != category.lower():
                        # Ensure it's actually a valid category
                        for cat in CATEGORIES:
                            print(cat)
                            if suggested_cat.lower() == cat.lower():
                                correct_category = cat
                                break
            except Exception as parse_err:
                print(f"[WARN] Could not parse category from reviewer output: {parse_err}")

            
                

            return {
                "status": "Rejected",
                "feedback": feedback,
                "correct_category": correct_category,
                "raw": output,
            }

        else:
            return {"status": "Rejected", "feedback": f"Unclear reviewer output: {output}", "raw": output}

    except Exception as e:
        print(f"[ERROR] Review failed: {e}")
        return {
            "status": "Rejected",
            "feedback": "Reviewer could not process request",
            "raw": str(e),
        }
