import streamlit as st
from agent.graph import graph, State, Context
from langgraph.runtime import Runtime


# Utility: run graph for a single ticket
def run_ticket(subject: str, description: str):
    # Create a new State object for each submission
    state = State(subject=subject, description=description)

    # Run the compiled graph with this fresh state
    final_state = graph.invoke(state)

    return final_state



# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Support Agent", page_icon="🤖", layout="centered")

st.title("🤖 AI Support Agent")
st.markdown("Submit a support ticket and let the AI draft a response!")

with st.form("ticket_form"):
    subject = st.text_input("Ticket Subject")
    description = st.text_area("Ticket Description", height=150)

    submitted = st.form_submit_button("Submit Ticket")

if submitted:
    if not subject or not description:
        st.error("⚠ Please enter both a subject and description.")
    else:
        with st.spinner("Processing your ticket..."):
            result_state = run_ticket(subject, description)

        st.subheader("📌 Ticket Classification")
        st.write(f"state: {result_state}")
        st.write(f"**Category:** {result_state["category"]}")
        st.write(f"**All Predictions:** {result_state["classification_scores"]}")

        st.subheader("📄 Draft Response")
        st.write(result_state["drafts"] or "No draft generated")

        st.subheader("✅ Review Result")
        st.write(f"**Status:** {result_state["review_status"]}")
        st.write(f"**Feedback:** {result_state["review_feedback"]}")

        if result_state["review_status"] and result_state["review_status"].lower() == "approved":
            st.success("🎉 Ticket approved automatically!")
        elif result_state["review_status"] and result_state["review_status"].lower() == "rejected":
            st.warning("⚠ Ticket was rejected — AI tried refining.")
        else:
            st.error("❌ Escalated to human support.")
