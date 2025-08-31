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


def review_response(draft: str, subject:str, desc:str) -> dict:
    """
    Review the draft response for compliance with support guidelines.
    Uses OpenRouter (Mistral-7B).
    
    Returns:
        {"status": "✅ Approved", "raw": "..."}
        or
        {"status": "❌ Rejected", "feedback": "...", "raw": "..."}
    """
    
    query=f"subject: {subject}, description:{desc}"
    prompt = f"""
You are a strict customer support quality assurance reviewer. 
You must reject the draft if it violates any of these rules:

❌ Reject if:
- It offers refunds, discounts, or financial commitments
- It promises something that support cannot guarantee (overpromising)
- It gives sensitive security advice (e.g., password resets, authentication bypass)
- It is rude, unprofessional, or unclear
- It is inaccurate or unhelpful
- It doesn't answer to user's query i.e gives irrelevant answer

✅ Approve if:
- It is accurate, helpful, polite, and compliant with the rules above.
-It is accurately answering to user's query: {query}

Here is the draft response:

--- DRAFT START ---
{draft}
--- DRAFT END ---

Respond with ONLY one of the following:
- "Approved"
- "Rejected: <feedback (reason for rejecting)>"
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
        print(output)

        cleaned = output.strip().lower().replace('"', '').strip()
        if cleaned.startswith("approved"):
            return {"status": "Approved", "raw": output}
        elif cleaned.startswith("rejected"):
            if ":" not in output:
                output = "Rejected: No reason provided"
            return {"status": "Rejected", "feedback": output, "raw": output}
        else:
            return {"status": "Rejected", "feedback": f"Unclear reviewer output: {output}", "raw": output}


    except Exception as e:
        print(f"[ERROR] Review failed: {e}")
        return {
            "status": "Rejected",
            "feedback": "Reviewer could not process request",
            "raw": str(e),
        }
