from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict
import asyncio
import os, csv

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
    category: dict = field(default_factory=dict)   # <-- now dictionary
    docs: list[str] = field(default_factory=list)
    drafts: dict = field(default_factory=dict)
    review_status: str = field(default=None)
    review_feedback: dict = field(default_factory=dict)
    attempts: int = 0
    category_changed: bool = False
    classification_scores: list = field(default_factory=list)  # store the ranking


# ---------------------- NODES ----------------------

def classify_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    # Reset for new submission
    state.attempts = 0
    state.review_status = None
    state.review_feedback = {}
    state.docs = []
    state.drafts = {}
    state.category_changed = False
    state.category = {}
    state.classification_scores = []

    print(state.subject, state.description)
    best_label, scores = classify_ticket(state.subject, state.description)
    print(f"[DEBUG] Classified as: {best_label}")
    print(f"[DEBUG] Classification scores: {scores}")

    # Save attempt 1 category
    state.category[state.attempts + 1] = best_label
    state.classification_scores = scores

    return {"category": state.category, "classification_scores": scores}


def retrieve_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    if not state.category:
        raise ValueError("Category is not set. Classification must run before retrieval.")

    current_cat = state.category[state.attempts + 1]

    docs = retrieve_context(
        current_cat,
        state.subject,
        state.description,
        top_k=3,
        attempt=state.attempts,
        classification_scores=state.classification_scores
    )
    state.docs = docs
    print(f"[DEBUG] Retrieved docs: {docs}")
    return {"docs": docs}


def draft_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    current_cat = state.category[state.attempts + 1]
    draft = generate_draft(state.subject, state.description, current_cat, state.docs)
    print(f"[DEBUG] Generated draft = {draft}")
    state.drafts[f"draft{state.attempts+1}"] = draft
    return {"drafts": state.drafts}


def review_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    if not state.drafts:
        raise ValueError("Draft is missing. Cannot run review.")

    draft_key = f"draft{state.attempts+1}"
    review_result = review_response(state.drafts[draft_key], state.subject, state.description)
    status = review_result.get("status", "Rejected")
    feedback = review_result.get("feedback", "No feedback provided.")

    state.review_status = status
    state.review_feedback[state.attempts+1] = feedback

    if status.lower().startswith("rejected"):
        state.attempts += 1
        print(f"rejected {state.attempts} times")

    return {
        "review_status": status,
        "review_feedback": state.review_feedback,
        "attempts": state.attempts,
        "category": state.category,
        "drafts": state.drafts
    }


def refine_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    print(f"[DEBUG] Refining query based on feedback: {state.review_feedback[state.attempts]}")

    # On the third attempt (attempts start at 0, so when == 2)
    if state.attempts == 2:
        scores = state.classification_scores
        if scores and len(scores) > 1:
            second_best = scores[1]
            print(f"[DEBUG] Switching category to second best: {second_best}")
            state.category[state.attempts + 1] = second_best
            state.category_changed = True
        else:
            state.category[state.attempts + 1] = state.category.get(state.attempts, "Unknown")
    else:
        # Keep same category as before
        prev_cat = state.category.get(state.attempts, None)
        state.category[state.attempts + 1] = prev_cat

    return {"category": state.category, "description": state.description}


def route_after_review(state: State) -> str:
    if state.review_status and state.review_status.lower() == "approved":
        state.attempts = 0
        return "__end__"
    if state.review_status and state.review_status.lower() == "rejected":
        if state.attempts < 3:
            return "refine_node"
        else:
            state.attempts = 0
            return "escalate_node"
    return "escalate_node"


def escalate_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    print("⚠ Escalating to human after 3 failed attempts.")
    state.attempts = 0
    escalation_data = {
        "subject": state.subject,
        "description": state.description,
        "final_category": state.category,
        "failed_drafts": "; ".join([f"{k}: {v}" for k, v in state.drafts.items()]),
        "review_feedbacks": "; ".join([f"Attempt {k}: {v}" for k, v in state.review_feedback.items()]),
    }
    file_exists = os.path.isfile("escalation_log.csv")
    with open("escalation_log.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "description",
                "final_category",
                "failed_drafts",
                "review_feedbacks",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(escalation_data)

    return {"review_status": "Escalated to human support"}


# ---------------------- GRAPH ----------------------

graph = (
    StateGraph(State, context_schema=Context)
    .add_node(classify_node)
    .add_node(retrieve_node)
    .add_node(draft_node)
    .add_node(review_node)
    .add_node(refine_node)
    .add_node(escalate_node)
    .add_edge("__start__", "classify_node")
    .add_edge("classify_node", "retrieve_node")
    .add_edge("retrieve_node", "draft_node")
    .add_edge("draft_node", "review_node")
    .add_conditional_edges("review_node", route_after_review)
    .add_edge("refine_node", "retrieve_node")
    .compile(name="Agent Graph")
)
