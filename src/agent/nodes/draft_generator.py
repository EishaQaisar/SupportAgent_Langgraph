from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def generate_draft(subject: str, description: str, category: str, docs: List[str]) -> str:
    

    context_summary = "\n".join([f"- {doc}" for doc in docs]) if docs else "No relevant context found."

    system_prompt = """
You are a professional and empathetic customer support agent. 
Your job is to directly answer the customer’s query in a short, clear, and polite way. 

Strict Guidelines:
- Always use the provided context (do not ignore any of it).
-Do NOT add ANY information by your own (use the context only)
-Even if context is wrong, just use it word to word
- Keep the response concise (3–6 sentences maximum).
- Do NOT include greetings (e.g., "Dear customer") or signatures (e.g., "Best regards").
- Do NOT say things like "As I mentioned before" — assume this is the first message.
- Write naturally as if you are replying directly with the solution.
"""

    user_prompt = f"""
Ticket Information:
- Category: {category}
- Subject: {subject}
- Description: {description}

Retrieved Context:
{context_summary}

Now, write the customer support reply.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",  
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=300,
    )

    draft = response.choices[0].message.content.strip()
    return draft
