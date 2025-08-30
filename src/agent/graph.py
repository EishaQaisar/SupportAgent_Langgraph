from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict
import asyncio

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from agent.nodes.classifier import classify_ticket
from agent.nodes.retriever import populate_db, retrieve_context
from agent.nodes.draft_generator import generate_draft
from agent.nodes.reviewer import review_response

# Pre-populate DB on startup
asyncio.run(asyncio.to_thread(populate_db))


class Context(TypedDict):
    """Context parameters for the agent."""
    pass


@dataclass
class State:
    subject: str
    description: str
    category: str = field(default=None)   # classification result
    docs: list[str] = field(default_factory=list)  # retrieved docs
    draft: str = field(default=None)      # generated draft
    review_status: str = field(default=None)  # Approved or Rejected
    review_feedback: str = field(default=None) # feedback if rejected
    attempts: int = 0  # 🔁 retry counter
    category_changed: bool = False


# ---------------------- NODES ----------------------

def classify_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    state.attempts=0
    print(state.description, state.subject)
    category =classify_ticket(state.subject, state.description)
    print(f"[DEBUG] Classified as: {category}")
    state.category = category
    return {"category": category}


def retrieve_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    if state.category is None:
        raise ValueError("Category is not set. Classification must run before retrieval.")
    
    docs = retrieve_context( state.category, state.subject, state.description,3, state.attempts,category_changed=state.category_changed)
    state.docs = docs
    state.category_changed=False
    print(f"[DEBUG] Retrieved docs: {docs}")
    return {"docs": docs}


def draft_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    draft = generate_draft( state.subject, state.description, state.category, state.docs)
    print(f"[DEBUG] Generated draft = {draft}")
    state.draft = draft
    return {"draft": draft}


def review_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    if not state.draft:
        raise ValueError("Draft is missing. Cannot run review.")
    print("state category",state.category)

    review_result = review_response(state.draft, state.subject, state.description, state.category)
    status = review_result.get("status", "Rejected")
    print("review status:", status)
    feedback = review_result.get("feedback", "No feedback provided.")
    correct_category = review_result.get("correct_category")
    print("complete result:", review_result)

    print(f"[DEBUG] Review result: {status}, feedback: {feedback}")

    # Update state with review outcome
    state.review_status = status
    state.review_feedback = feedback

  

    # Increment attempts only on rejection
    if status.lower() == "rejected":
        if correct_category!= state.category:
            print("current category is", state.category)
            print("correct category is", correct_category)
            print(f"[DEBUG] Updating category from '{state.category}' to '{correct_category}'")
            state.category = correct_category
            state.category_changed = True
            print("updated category:", state.category)
        else:
            state.category_changed = False
        print(f"rejected {state.attempts} times")
        state.attempts += 1
    print(state.attempts)

    return {
        "review_status": status,
        "review_feedback": feedback,
        "correct_category": correct_category,
        "attempts": state.attempts,
        "category": state.category,   # include updated category in return
    }

def refine_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    print(f"[DEBUG] Refining query based on feedback: {state.review_feedback}")

  

    # Make description more specific
    #state.description += f"\n(Reviewer hint: {state.review_feedback})"

    return {"category": state.category, "description": state.description}


def route_after_review(state: State) -> str:
    if state.review_status and state.review_status.lower() == "approved":
        state.attempts=0
        print("ATTEMPTS RESET")
        return "__end__"
    if state.review_status and state.review_status.lower() == "rejected":
        if state.attempts < 3:   # allow up to 3 total attempts
            return "refine_node"
        else:
            state.attempts=0
            return "escalate_node"
    return "escalate_node"  # fallback safety


def escalate_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    print("⚠ Escalating to human after 3 failed attempts.")
    state.attempts=0
    print("Attempts reet for next ticket", state.attempts)
    return {"final": "Escalated to human support"}


# ---------------------- GRAPH ----------------------

graph = (
    StateGraph(State, context_schema=Context)
    .add_node(classify_node)
    .add_node(retrieve_node)
    .add_node(draft_node)
    .add_node(review_node)   # reviewer node
    .add_node(refine_node)
    .add_node(escalate_node)
    .add_edge("__start__", "classify_node")
    .add_edge("classify_node", "retrieve_node")
    .add_edge("retrieve_node", "draft_node")
    .add_edge("draft_node", "review_node")   # review after draft
    .add_conditional_edges("review_node", route_after_review)
    .add_edge("refine_node", "retrieve_node")  # loop back for retry
    .compile(name="Agent Graph")
)
