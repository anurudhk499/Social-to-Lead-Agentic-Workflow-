"""
graph.py
--------
LangGraph-based Conversational AI Agent for AutoStream.

Graph nodes:
    1. classify_intent  → Determine user intent
    2. rag_retrieve     → Fetch relevant KB context
    3. collect_lead     → Parse / request lead info
    4. capture_lead     → Call mock_lead_capture tool
    5. generate_response → Produce final reply

State machine:
    START
      └─► classify_intent
            ├─► [greeting / chitchat]   → generate_response → END
            ├─► [product_inquiry]       → rag_retrieve → generate_response → END
            └─► [high_intent]           → collect_lead
                    ├─► [incomplete]    → generate_response → END
                    └─► [complete]      → capture_lead → generate_response → END
"""

import os
import re
from typing import TypedDict, Optional, Literal, Annotated
from operator import add


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from agent.rag_pipeline import RAGRetriever
from agent.tools import mock_lead_capture


# ─────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Full conversation history (HumanMessage / AIMessage objects)
    messages: Annotated[list, add]

    # Detected intent: "greeting" | "product_inquiry" | "high_intent"
    intent: str

    # Lead collection fields (None until provided by user)
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Next field the agent is currently asking for
    awaiting_field: Optional[str]   # "name" | "email" | "platform" | None

    # Whether lead has been successfully captured
    lead_captured: bool

    # RAG context retrieved for the current turn
    rag_context: str

    # The agent's response text for the current turn
    response_text: str


# ─────────────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────────────

def get_llm():
    import os
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=api_key,
    )


# ─────────────────────────────────────────────────────────────────
# HELPER – extract last user message text
# ─────────────────────────────────────────────────────────────────

def last_user_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ─────────────────────────────────────────────────────────────────
# NODE 1 – classify_intent
# ─────────────────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """Use the LLM to classify the latest user message into one of three intents."""
    llm = get_llm()
    user_text = last_user_text(state)

    # If we are already in lead-collection mode, keep intent as high_intent
    if state.get("awaiting_field") or (
        state.get("lead_name") and not state.get("lead_captured")
    ):
        return {**state, "intent": "high_intent"}

    system_prompt = """You are an intent classifier for AutoStream, a SaaS video editing product.

Classify the user message into EXACTLY one of these labels:
  - greeting        : casual hello, small talk, generic opener
  - product_inquiry : asking about features, pricing, plans, policies, FAQs
  - high_intent     : ready to sign up, wants to try, wants to buy, expresses strong interest

Respond with ONLY the label, nothing else."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    intent = response.content.strip().lower()

    # Normalise to valid labels
    if intent not in ("greeting", "product_inquiry", "high_intent"):
        intent = "product_inquiry"   # safe fallback

    return {**state, "intent": intent}


# ─────────────────────────────────────────────────────────────────
# NODE 2 – rag_retrieve
# ─────────────────────────────────────────────────────────────────

_retriever: Optional[RAGRetriever] = None   # module-level singleton

def rag_retrieve(state: AgentState) -> AgentState:
    """Retrieve the most relevant KB chunks for the user's query."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(k=3)

    query = last_user_text(state)
    context = _retriever.retrieve(query)
    return {**state, "rag_context": context}


# ─────────────────────────────────────────────────────────────────
# NODE 3 – collect_lead
# ─────────────────────────────────────────────────────────────────

def collect_lead(state: AgentState) -> AgentState:
    """
    Parse the latest user message for lead info.
    Determine which field to ask for next.
    """
    user_text = last_user_text(state).strip()

    lead_name     = state.get("lead_name")
    lead_email    = state.get("lead_email")
    lead_platform = state.get("lead_platform")
    awaiting      = state.get("awaiting_field")

    # ── Fill in whichever field we were awaiting ──────────────────
    if awaiting == "name" and not lead_name:
        lead_name = user_text.title()

    elif awaiting == "email" and not lead_email:
        # Basic email validation
        if re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", user_text):
            lead_email = re.search(
                r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", user_text
            ).group()
        else:
            # Not a valid email – keep awaiting email
            return {
                **state,
                "response_text": (
                    "That doesn't look like a valid email. "
                    "Could you please double-check and re-enter your email address?"
                ),
                "awaiting_field": "email",
            }

    elif awaiting == "platform" and not lead_platform:
        # Accept common platforms or whatever the user types
        known = ["youtube", "instagram", "tiktok", "twitter", "facebook", "linkedin"]
        lower = user_text.lower()
        for p in known:
            if p in lower:
                lead_platform = p.capitalize()
                break
        if not lead_platform:
            lead_platform = user_text.title()   # accept custom input

    # ── Decide which field to ask for next ───────────────────────
    if not lead_name:
        next_field = "name"
    elif not lead_email:
        next_field = "email"
    elif not lead_platform:
        next_field = "platform"
    else:
        next_field = None   # all collected → ready to capture

    return {
        **state,
        "lead_name":     lead_name,
        "lead_email":    lead_email,
        "lead_platform": lead_platform,
        "awaiting_field": next_field,
    }


# ─────────────────────────────────────────────────────────────────
# NODE 4 – capture_lead
# ─────────────────────────────────────────────────────────────────

def capture_lead(state: AgentState) -> AgentState:
    """Call mock_lead_capture once all lead fields are collected."""
    mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"],
    )
    return {**state, "lead_captured": True}


# ─────────────────────────────────────────────────────────────────
# NODE 5 – generate_response
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Alex, a friendly sales assistant for AutoStream – 
an AI-powered video editing SaaS for content creators.

Your goals:
1. Answer questions accurately using only the provided knowledge base context.
2. Be warm, concise, and helpful.
3. When collecting lead info, ask for ONE field at a time (name, then email, then platform).
4. Never make up features or prices not in the context.
5. Keep responses under 120 words unless a detailed comparison is needed.
"""

def generate_response(state: AgentState) -> AgentState:
    """Generate the agent's natural language reply."""
    llm = get_llm()

    # If a response was already set by collect_lead (validation error), use it
    if state.get("response_text") and state.get("awaiting_field") == "email":
        return state

    intent         = state.get("intent", "greeting")
    rag_context    = state.get("rag_context", "")
    lead_name      = state.get("lead_name")
    lead_email     = state.get("lead_email")
    lead_platform  = state.get("lead_platform")
    awaiting       = state.get("awaiting_field")
    lead_captured  = state.get("lead_captured", False)
    user_text      = last_user_text(state)

    # ── Build context block ───────────────────────────────────────
    context_block = ""
    if rag_context:
        context_block = f"\n\nKNOWLEDGE BASE CONTEXT:\n{rag_context}"

    # ── Build lead-status block ───────────────────────────────────
    lead_status = ""
    if intent == "high_intent":
        collected = []
        if lead_name:     collected.append(f"name={lead_name}")
        if lead_email:    collected.append(f"email={lead_email}")
        if lead_platform: collected.append(f"platform={lead_platform}")
        lead_status = f"\n\nLEAD COLLECTION STATUS: {', '.join(collected) if collected else 'none collected yet'}"
        if awaiting:
            lead_status += f"\nNEXT FIELD TO ASK: {awaiting}"
        if lead_captured:
            lead_status += "\nLEAD ALREADY CAPTURED – thank the user and wrap up."

    # ── Build instruction for this turn ──────────────────────────
    if lead_captured:
        instruction = (
            "The lead has been captured successfully. "
            "Thank the user warmly, confirm you'll be in touch soon, "
            "and offer to answer any remaining questions."
        )
    elif intent == "high_intent" and awaiting:
        field_prompts = {
            "name":     "Acknowledge their interest and ask for their full name.",
            "email":    f"Thank them for their name ({lead_name}). Now ask for their email address.",
            "platform": f"Great, you have their name and email. Now ask which creator platform they use (e.g. YouTube, Instagram, TikTok).",
        }
        instruction = field_prompts.get(awaiting, "Continue collecting lead information.")
    elif intent == "greeting":
        instruction = "Greet the user warmly, introduce AutoStream briefly, and invite them to ask questions."
    else:
        instruction = "Answer the user's question using the knowledge base context provided. Be accurate and concise."

    # ── Compose history for multi-turn memory ─────────────────────
    history = []
    for msg in state["messages"][:-1]:   # exclude last (current user turn)
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    prompt = (
        f"{SYSTEM_PROMPT}"
        f"{context_block}"
        f"{lead_status}"
        f"\n\nINSTRUCTION FOR THIS TURN: {instruction}"
        f"\n\nUser said: \"{user_text}\""
    )

    messages_to_send = [SystemMessage(content=prompt)]
    for h in history[-6:]:   # keep last 6 turns for context window efficiency
        if h["role"] == "user":
            messages_to_send.append(HumanMessage(content=h["content"]))
        else:
            messages_to_send.append(AIMessage(content=h["content"]))
    messages_to_send.append(HumanMessage(content=user_text))

    reply = llm.invoke(messages_to_send)
    return {**state, "response_text": reply.content.strip()}


# ─────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS (conditional edges)
# ─────────────────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> Literal[
    "rag_retrieve", "collect_lead", "generate_response"
]:
    intent = state.get("intent", "greeting")
    if intent == "greeting":
        return "generate_response"
    elif intent == "product_inquiry":
        return "rag_retrieve"
    else:   # high_intent
        return "collect_lead"


def route_after_collect(state: AgentState) -> Literal["capture_lead", "generate_response"]:
    awaiting = state.get("awaiting_field")
    if awaiting is None:   # all three fields collected
        return "capture_lead"
    return "generate_response"


# ─────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent",   classify_intent)
    graph.add_node("rag_retrieve",      rag_retrieve)
    graph.add_node("collect_lead",      collect_lead)
    graph.add_node("capture_lead",      capture_lead)
    graph.add_node("generate_response", generate_response)

    # Entry point
    graph.add_edge(START, "classify_intent")

    # After intent → branch
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "generate_response": "generate_response",
            "rag_retrieve":      "rag_retrieve",
            "collect_lead":      "collect_lead",
        },
    )

    # RAG → always goes to response
    graph.add_edge("rag_retrieve", "generate_response")

    # After collect → branch on completeness
    graph.add_conditional_edges(
        "collect_lead",
        route_after_collect,
        {
            "capture_lead":      "capture_lead",
            "generate_response": "generate_response",
        },
    )

    # Capture → always goes to response
    graph.add_edge("capture_lead", "generate_response")

    # Response → always ends
    graph.add_edge("generate_response", END)

    return graph.compile()
