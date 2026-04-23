# 🎬 AutoStream AI Sales Agent

> A Conversational AI Agent for AutoStream — an AI-powered video editing SaaS for content creators.  
> Built for the **ServiceHive Inflx ML Intern Assignment**.

---

## 📁 Project Structure

```
autostream-agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph state machine (core agent logic)
│   ├── rag_pipeline.py   # RAG pipeline (FAISS + HuggingFace embeddings)
│   └── tools.py          # Lead capture tool (mock API)
├── knowledge_base/
│   └── autostream_kb.json  # Pricing, policies, FAQs
├── main.py               # CLI entry point
├── .env                  # API key (create this yourself, do not commit)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- A free Groq API key from [console.groq.com](https://console.groq.com/)

### Step 1 – Clone / download the project

```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### Step 2 – Create and activate a virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

> If you get a scripts error on Windows, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB). This is a one-time download.

### Step 4 – Set your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your-key-here
```

Get your free key at [console.groq.com](https://console.groq.com/) → API Keys → Create API Key. No credit card required.

### Step 5 – Run the agent

**Interactive chat (recommended for demo):**
```bash
python main.py
```

**Automated scripted demo:**
```bash
python main.py --demo
```

> 💡 Every time you open a new terminal, activate the venv first:
> ```powershell
> .\venv\Scripts\Activate.ps1
> ```
> The `.env` file handles the API key automatically.

---

## 🧠 Architecture Explanation (~200 words)

The agent is built using **LangGraph**, chosen over AutoGen because LangGraph offers explicit, inspectable state transitions via a directed graph — making it ideal for a multi-step agentic workflow where control flow (greeting → inquiry → lead collection → capture) must be deterministic and auditable. AutoGen is powerful for multi-agent collaboration but adds unnecessary complexity for a single-agent pipeline with defined transitions.

### State Management
The entire conversation context is stored in a typed `AgentState` dictionary that persists across all graph nodes within a session. It tracks: the full message history (for multi-turn memory), classified intent, three lead fields (`lead_name`, `lead_email`, `lead_platform`), which field is currently `awaiting`, and whether the lead has been `captured`. The graph passes this state through every node, with each node returning an updated copy — making state transitions explicit and debuggable.

### Graph Flow
```
START → classify_intent
              ├─ greeting        → generate_response
              ├─ product_inquiry → rag_retrieve → generate_response
              └─ high_intent     → collect_lead
                                      ├─ incomplete → generate_response
                                      └─ complete   → capture_lead → generate_response → END
```

**RAG** uses `sentence-transformers/all-MiniLM-L6-v2` for local embeddings and FAISS for vector similarity search over the knowledge base JSON — no external API calls for retrieval.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API** (Meta) with a webhook-based architecture:

### Architecture

```
WhatsApp User
     │  (sends message)
     ▼
Meta Cloud API
     │  (POST webhook event)
     ▼
FastAPI / Flask Server  ←─── Your hosted backend
     │
     ├── Verify webhook (GET, X-Hub-Signature-256)
     │
     └── On POST:
           1. Parse incoming message (from, body, message_id)
           2. Load session state from Redis (keyed by phone number)
           3. Run LangGraph agent with user message
           4. Save updated state back to Redis
           5. POST reply to WhatsApp Send Message API
           6. Return HTTP 200 to Meta
```

### Key Implementation Steps

1. **Register a Meta App** at [developers.facebook.com](https://developers.facebook.com/) and get a WhatsApp Business number.

2. **Create a webhook endpoint** (FastAPI example):

```python
from fastapi import FastAPI, Request
import httpx, redis, json

app = FastAPI()
r = redis.Redis()

WHATSAPP_TOKEN = "your-meta-token"
PHONE_NUMBER_ID = "your-phone-id"

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == "MY_VERIFY_TOKEN":
        return int(params["hub.challenge"])
    return 403

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    msg_data = body["entry"][0]["changes"][0]["value"]

    if "messages" not in msg_data:
        return {"status": "ok"}

    msg = msg_data["messages"][0]
    phone = msg["from"]
    user_text = msg["text"]["body"]

    # Load state from Redis (session persistence across turns)
    raw = r.get(f"session:{phone}")
    state = json.loads(raw) if raw else initial_state()

    # Run agent
    state["messages"].append({"role": "user", "content": user_text})
    state = graph.invoke(state)
    reply = state["response_text"]

    # Save state
    r.setex(f"session:{phone}", 3600, json.dumps(state))

    # Send reply via WhatsApp API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": phone,
                "text": {"body": reply}
            }
        )
    return {"status": "ok"}
```

3. **Deploy** the FastAPI server on **Railway / Render / EC2** with a public HTTPS URL.
4. **Register the webhook URL** in the Meta Developer Dashboard.
5. **Session state** is stored in Redis keyed by phone number, so each user maintains independent multi-turn memory.

---

## 🛠️ Tech Stack

| Component        | Technology                               |
|------------------|------------------------------------------|
| Agent Framework  | LangGraph 0.2+                           |
| LLM              | LLaMA 3.1 8B Instant via Groq API (free) |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2   |
| Vector Store     | FAISS (local, CPU)                       |
| Language         | Python 3.10+                             |
| WhatsApp Deploy  | Meta Cloud API + FastAPI + Redis         |

---

## 📋 Evaluation Checklist

- [x] Intent classification (greeting / product inquiry / high intent)
- [x] RAG-powered knowledge retrieval from local JSON
- [x] State retention across 5–6+ conversation turns
- [x] Tool execution only after all 3 lead fields are collected
- [x] Email validation before accepting
- [x] `mock_lead_capture()` prints lead ID, name, email, platform
- [x] Demo mode for scripted walkthrough
- [x] Interactive chat mode for live demo

---

*Built by Anurudh Sharma | ServiceHive ML Intern Assignment*
