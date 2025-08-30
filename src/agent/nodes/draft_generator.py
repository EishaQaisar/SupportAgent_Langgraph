# src/agent/nodes/draft_generator.py
from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Init OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def generate_draft(subject: str, description: str, category: str, docs: List[str]) -> str:
    """
    Generate an initial draft response for the user ticket using an LLM.
    Combines ticket details + retrieved docs into a clear, customer-friendly draft.
    """

    context_summary = "\n".join([f"- {doc}" for doc in docs]) if docs else "No relevant context found."

    system_prompt = """
You are a professional and empathetic customer support agent. 
Your job is to directly answer the customer’s query in a short, clear, and polite way. 

Guidelines:
- Always use the provided context to give the answer (do not ignore it).
- Keep the response concise (3–6 sentences maximum).
- Do NOT include greetings like "Dear customer" or signatures such as "Best regards".
- Do NOT repeat the subject line.
- Write naturally as if you are the agent replying directly.
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

    draft=context_summary
    return draft
