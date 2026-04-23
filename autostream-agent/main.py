"""
main.py
-------
CLI entry point for the AutoStream Conversational AI Agent.

Run:
    python main.py

Requirements:
    export ANTHROPIC_API_KEY="your-key-here"
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import build_graph, AgentState


# ─────────────────────────────────────────────────────────────────
# INITIAL STATE
# ─────────────────────────────────────────────────────────────────

def initial_state() -> AgentState:
    return {
        "messages":      [],
        "intent":        "greeting",
        "lead_name":     None,
        "lead_email":    None,
        "lead_platform": None,
        "awaiting_field": None,
        "lead_captured": False,
        "rag_context":   "",
        "response_text": "",
    }


# ─────────────────────────────────────────────────────────────────
# CHAT LOOP
# ─────────────────────────────────────────────────────────────────

def run_chat():
    print("\n" + "=" * 60)
    print("  🎬  AutoStream AI Sales Agent  (powered by Claude Haiku)")
    print("=" * 60)
    print("  Type 'quit' or 'exit' to end the conversation.\n")

    graph = build_graph()
    state = initial_state()

    while True:
        # ── Get user input ─────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\nAgent: Thanks for chatting! Have a great day! 🎬")
            break

        # ── Append user message to state ───────────────────────────
        state = {
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_input)],
            "rag_context":   "",
            "response_text": "",
        }

        # ── Run through the graph ──────────────────────────────────
        try:
            state = graph.invoke(state)
        except Exception as e:
            print(f"\n⚠️  Agent error: {e}\n")
            continue

        # ── Print response and append to history ───────────────────
        reply = state.get("response_text", "I'm sorry, I couldn't generate a response.")
        print(f"\nAgent: {reply}\n")

        state = {
            **state,
            "messages": state["messages"] + [AIMessage(content=reply)],
        }

        # ── Stop if lead was just captured ─────────────────────────
        if state.get("lead_captured"):
            print("─" * 60)
            print("✅  Lead capture complete. Session ended.")
            print("─" * 60)
            break


# ─────────────────────────────────────────────────────────────────
# DEMO MODE – runs a scripted conversation automatically
# ─────────────────────────────────────────────────────────────────

def run_demo():
    """
    Simulates the exact conversation flow specified in the assignment.
    Use: python main.py --demo
    """
    demo_inputs = [
        "Hi, tell me about your pricing.",
        "What's included in the Pro plan?",
        "That sounds great, I want to sign up for the Pro plan for my YouTube channel.",
        "Anurudh Sharma",
        "anurudh@example.com",
        "YouTube",
    ]

    print("\n" + "=" * 60)
    print("  🎬  AutoStream Agent — DEMO MODE")
    print("=" * 60)
    print("  Running scripted demo conversation...\n")

    graph = build_graph()
    state = initial_state()

    for user_input in demo_inputs:
        print(f"You: {user_input}")

        state = {
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_input)],
            "rag_context":   "",
            "response_text": "",
        }

        state = graph.invoke(state)
        reply = state.get("response_text", "")
        print(f"Agent: {reply}\n")

        state = {
            **state,
            "messages": state["messages"] + [AIMessage(content=reply)],
        }

        if state.get("lead_captured"):
            print("─" * 60)
            print("✅  Demo complete. Lead captured successfully.")
            print("─" * 60)
            break


# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_chat()
