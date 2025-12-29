"""Streamlit app
Run:
  streamlit run streamlit_app.py
"""

import streamlit as st

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from langchain.schema import HumanMessage

from document_loader import DocumentLoader
from rag import graph, config, retriever

# Page config
st.set_page_config(page_title="Marketing Operations AI Assistant (Agentic Prototype)", layout="wide")
st.title("📚 Marketing Operations AI Assistant (Agentic Prototype)")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# ---- Tabs ----
tab_chat, tab_about, tab_howto, tab_arch = st.tabs(
    ["💬 Chat", "ℹ️ About this app", "🧭 How to use / guideline", "🏗️ Architecture & Technical Details"]
)

# =========================
# TAB 1: CHAT
# =========================
with tab_chat:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat Interface")

        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            with st.chat_message(role):
                st.markdown(content)

        user_input = st.chat_input("Ask a question about your documents...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if not st.session_state.rag_ready:
                assistant_response = "Please upload documents first, then click **Build Knowledge Base**."
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = graph.invoke(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                        )
                        final_message = result["messages"][-1].content
                        st.markdown(final_message)

                st.session_state.chat_history.append({"role": "assistant", "content": final_message})

    with col2:
        st.subheader("Document Management")

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=list(DocumentLoader.supported_extensions),
            accept_multiple_files=True,
        )

        if uploaded_files:
            existing_names = {f.name for f in st.session_state.uploaded_files}
            for file in uploaded_files:
                if file.name not in existing_names:
                    st.session_state.uploaded_files.append(file)

        st.write("### Uploaded Files")
        if st.session_state.uploaded_files:
            for f in st.session_state.uploaded_files:
                st.write(f"- {f.name}")
        else:
            st.info("No files uploaded yet.")

        if st.button("Build Knowledge Base"):
            if not st.session_state.uploaded_files:
                st.warning("Please upload at least one document first.")
            else:
                with st.spinner("Indexing documents..."):
                    retriever.add_documents_from_uploads(st.session_state.uploaded_files)
                    st.session_state.rag_ready = True
                st.success("Knowledge base is ready! You can now ask questions.")

        if st.button("Clear Session"):
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            st.session_state.rag_ready = False
            st.success("Session cleared.")

# =========================
# TAB 2: ABOUT
# =========================
with tab_about:
    st.header("About this app")

    st.markdown(
        """
This application is a **Marketing Operations AI Assistant (Agentic Prototype)** designed to help Marketing teams work faster, more consistently, and with greater confidence by grounding AI responses in **our internal documentation, standards, and operating model**.

It is **not a replacement for Microsoft Copilot**.  
Instead, it complements Copilot by focusing on **marketing-specific knowledge and workflows** that generic enterprise AI tools do not reliably capture.

---

### Why this matters

Marketing organizations often struggle with:
- Inconsistent definitions (KPIs, attribution, lifecycle stages)
- Knowledge spread across decks, PDFs, playbooks, and SOPs
- Repeated questions about “how we do things here”
- Time lost translating strategy into execution details

This assistant addresses those gaps by acting as a **single, trusted intelligence layer** over approved Marketing content.

---

### What this assistant does today

- Ingests internal marketing documents (guidelines, frameworks, playbooks)
- Retrieves the most relevant content based on user questions
- Generates answers **grounded in internal sources**, reducing hallucination
- Helps users:
  - Clarify definitions and operating rules
  - Summarize policies and standards
  - Draft internal-ready artifacts (briefs, checklists, SOPs)

This prototype demonstrates the **core agentic loop**:  
**Retrieve → Reason → Respond**, using our content as the source of truth.

---

### How this complements Microsoft Copilot

Microsoft Copilot excels at:
- General productivity
- Cross-tool assistance (Outlook, Teams, PowerPoint, Excel)
- Broad language understanding

This assistant focuses on:
- Marketing-specific standards and definitions
- Operational consistency and governance
- Domain-aware reasoning tied to how Marketing actually operates

Together, they form a layered model:
- **Copilot** → general enterprise productivity  
- **Marketing Operations AI Assistant** → domain-specific enablement and intelligence

---

### What this can evolve into

With further investment, this prototype can mature into a daily-use **Agentic Marketing Enablement Tool**, capable of:
- Assisting campaign setup and QA
- Enforcing standards at execution time
- Supporting analytics interpretation and experimentation design
- Powering guided workflows (e.g., “create a campaign brief”, “validate a launch checklist”)
- Integrating with MarTech, analytics, and data platforms

---

### Important note

This is an **exploratory, internal prototype** intended to demonstrate feasibility, collect stakeholder feedback, and inform a broader Agentic AI strategy for Marketing.

We evaluate this like an internal operations tool, not a chatbot.
It’s reliable because it’s grounded in our own documents, consistent because it enforces our definitions, and useful because it reduces time spent searching and clarifying.
The prototype proves the core loop works. The next step is governance, integration, and scale.

- Enterprise-grade governance, access control, persistence, and integrations would be addressed prior to any production deployment.
        """
    )

# =========================
# TAB 3: HOW TO USE
# =========================
with tab_howto:
    st.header("How to use / guideline")

    st.markdown(
        """
### Quick start
1. **Upload** your docs (PDF, TXT, DOCX, etc.)
2. Click **Build Knowledge Base**
3. Ask questions in the chat

### Best practices (for stronger answers)
- Ask **specific questions** (e.g., “What is our definition of MQL?” vs “Explain MQL”)
- Provide context in your question: channel, market, product line, timeframe
- If you want a specific output format, request it (e.g., “Answer as a checklist”)

### Example questions
- “Summarize our modern marketing operations framework in 10 bullets.”
- “What does our guideline say about attribution models and limitations?”
- “Draft a campaign brief template aligned with our definitions.”
- “Create a QA checklist for email + paid social launch based on our standards.”

### Limitations (current prototype)
- The knowledge base is built from **uploaded docs only** (not yet connected to Drive/Confluence/Slack)
- Citations are not shown as clickable sources yet (can be added next)
- Index is stored in memory per session (can be upgraded to persistent storage)

### Data handling note
Avoid uploading restricted or highly sensitive documents until we add governance controls:
access control, audit logs, encryption, and retention policies.
        """
    )

# =========================
# TAB 4: ARCHITECTURE
# =========================
with tab_arch:
    st.header("Architecture & Technical Details")

    st.markdown(
        """
This application is built using a **modular, agentic architecture** designed for clarity, scalability, and enterprise evolution.

Think of the system as **four layers**, moving from user interaction to intelligent response.

---

### 1. User Interaction Layer (Front End)

**What happens here**
- Users upload approved Marketing documents
- Users ask questions in natural language
- Responses are displayed in a chat-style interface

**Why it matters**
- Simple, intuitive experience
- No training required for business users
- Designed for daily, repeat use

---

### 2. Knowledge Ingestion & Indexing Layer

**What happens here**
- Uploaded documents are parsed and cleaned
- Long documents are split into smaller, meaningful sections
- Each section is converted into a numerical representation (embedding)
- These embeddings are stored in a searchable index

**Why it matters**
- Allows the system to find the *right* information quickly
- Prevents the AI from relying on generic knowledge
- Keeps answers grounded in approved internal content

---

### 3. Agentic Reasoning Layer (RAG Pipeline)

This is the intelligence core of the system.

**Step-by-step flow**
1. **Retrieve**  
   The system identifies the most relevant content from internal documents based on the user’s question.

2. **Reason**  
   A large language model interprets the question using both the user input and retrieved context.

3. **Respond**  
   The system generates a clear, actionable answer aligned with Marketing standards.

**Why it matters**
- Reduces hallucination risk
- Improves trust and adoption
- Enables future multi-step, tool-using agents

---

### 4. Language Model Layer

**What happens here**
- A high-performance LLM generates responses
- The model is constrained by retrieved internal context
- Outputs are shaped for business readability

**Why it matters**
- Fast, conversational experience
- Flexibility to swap or upgrade models over time
- Cost and performance optimization options

---

### Current prototype vs. production evolution

**Today (Prototype)**
- In-memory knowledge index
- Session-based usage
- Manual document upload

**Future (Production-ready)**
- Persistent vector storage
- Role-based access control
- Source citations and traceability
- Integration with enterprise systems (Drive, Confluence, SharePoint, MarTech tools)
- Observability (usage, cost, quality metrics)

---

### Why this architecture was chosen

- Modular: each layer can evolve independently
- Enterprise-aligned: supports governance and scale
- Agent-ready: designed to expand beyond Q&A into workflow automation

This architecture provides a **safe, flexible foundation** for introducing Agentic AI into Marketing operations without disrupting existing enterprise platforms.

        """
    )

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <hr style="margin-top:32px;margin-bottom:8px;">
    <div style="text-align:center;font-size:12px;color:gray;">
      © 2025 Howard Nguyen, PhD. For prototype and demonstration only.
    </div>
    """,
    unsafe_allow_html=True
)