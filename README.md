
# 🛠️ AI Support Agent with LangGraph

An AI-powered **support ticket resolution agent** built using **LangGraph**, **RAG (FAISS + SentenceTransformers)**, and **LLMs**.  
The agent can classify support tickets, retrieve relevant knowledge base docs, generate draft responses, and review them for compliance.

---

## 🚀 Features
- **Ticket Classification** → Categorizes tickets into domains (Billing, Technical, General, Security).  
- **Context Retrieval (RAG)** → Retrieves semantically similar docs using FAISS + embeddings.  
- **Draft Generation (LLM: Mistral-7B)** → Generates context-aware responses with grounding in knowledge base.  
- **Policy Review (LLM: Mistral-7B)** → Reviewer node ensures compliance and prevents invalid promises.  
- **Feedback Loop** → Reviewer feedback used to retry generation (up to 2 attempts).  
- **Streamlit UI** → Interactive demo interface for testing the agent.  

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-support-agent.git
cd ai-support-agent
 ```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash

pip install -r requirements.txt
```

### 4. Environment Variables

Create a .env file in the project root with the following keys:

```bash
OPENROUTER_API_KEY=your_openrouter_key
LANGSMITH_API_KEY=your_langsmith_key
```

### ▶️ Running the Agent
Option 1: LangGraph Dev Server
```bash

langgraph dev
```

This launches the LangGraph API server. You can interact with the agent via API calls (e.g., Postman, curl).

Option 2: Streamlit App
```bash

streamlit run app.py
```


This launches an interactive web UI for testing tickets.

## 🧪 Testing the Agent

Submit tickets like:

"I was charged twice on my bill" → should classify as Billing.

"My password reset link isn’t working" → should classify as Security.

Reviewer node will approve ✅ relevant, policy-compliant responses.

Irrelevant or invalid responses will be rejected ❌ and retried (up to 2 times).

## 🏗️ Design & Architectural Decisions

The support agent is built as a LangGraph workflow with modular nodes:

#### 1. LangGraph Orchestration

Defines the workflow as a stateful graph.

Steps: Classifier → Retriever → Draft Generator → Reviewer.

#### 2. Classifier (facebook/bart-large-mnli)

Classifies tickets into categories like Billing, Technical, General, Security.

Uses facebook/bart-large-mnli, a strong NLI-based zero-shot classifier.

#### 3. Retriever (FAISS + SentenceTransformers)

Embeds knowledge base docs using all-MiniLM-L6-v2.

Stores embeddings in FAISS for fast semantic similarity search.

Returns top-k relevant docs for grounding.

#### 4. Draft Generator (Mistral-7B via OpenRouter)

Generates initial response drafts using Mistral-7B (served via OpenRouter).

Inputs: ticket, category, retrieved docs.

Produces fluent, grounded draft responses.

#### 5. Reviewer (Mistral-7B via OpenRouter)

Uses the same Mistral-7B LLM to evaluate draft quality.

#### Validates:

Accuracy

Professional tone

Compliance (no refunds/promises without approval)

#### 6. Feedback Loop

Reviewer rejection triggers refinement:

Feedback is injected back into retrieval + generation.

Draft regeneration occurs (max 2 attempts).

#### 7. Streamlit UI

Simple interface for testing without Postman/curl.

Shows ticket → retrieved docs → draft → review result.

## 📦 Requirements

All dependencies are listed in requirements.txt.
Key libraries include:

langgraph

openai

transformers

sentence-transformers

faiss-cpu

numpy

python-dotenv

streamlit

## 📌 Extensibility

Swap classifier (e.g., upgrade from bart-large-mnli to LLaMA-2).

Swap draft generator/reviewer model (e.g., Mistral → GPT-4 → Claude).

Add more reviewer rules or integrate human-in-the-loop.

Expand categories and knowledge base easily.

## 🖼️ Workflow Diagram
```mermaid

flowchart TD
    A["Ticket"] --> B["Classifier: bart-large-mnli"]
    B --> C["Retriever: FAISS + SentenceTransformers"]
    C --> D["Draft Generator: Mistral-7B"]
    D --> E["Reviewer: Mistral-7B"]
    E --✅ Approved--> F["Final Response"]
    E --❌ Rejected--> C


