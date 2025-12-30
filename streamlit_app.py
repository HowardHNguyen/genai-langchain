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


# =========================
# Page config
# =========================
st.set_page_config(page_title="Marketing Operations AI Assistant (Agentic Prototype)", layout="wide")
st.title("📚 Marketing Operations AI Assistant (Agentic Prototype)")


# =========================
# Session state
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# =========================
# Tabs
# =========================
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
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about your documents...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if not st.session_state.rag_ready:
                reply = "Please upload documents first, then click **Build Knowledge Base**."
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = graph.invoke(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                        )
                        answer = result["messages"][-1].content
                        st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with col2:
        st.subheader("Document Management")

        uploads = st.file_uploader(
            "Upload Documents",
            type=list(DocumentLoader.supported_extensions),
            accept_multiple_files=True,
        )

        if uploads:
            existing = {f.name for f in st.session_state.uploaded_files}
            for f in uploads:
                if f.name not in existing:
                    st.session_state.uploaded_files.append(f)

        st.write("### Uploaded Files")
        if st.session_state.uploaded_files:
            for f in st.session_state.uploaded_files:
                st.write(f"- {f.name}")
        else:
            st.info("No files uploaded yet.")

        if st.button("Build Knowledge Base"):
            if not st.session_state.uploaded_files:
                st.warning("Please upload documents first.")
            else:
                with st.spinner("Indexing documents..."):
                    retriever.add_documents_from_uploads(st.session_state.uploaded_files)
                    st.session_state.rag_ready = True
                st.success("Knowledge base ready.")

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

We evaluate this like an internal operations tool, not a chatbot. It’s reliable because it’s grounded in our own documents, consistent because it enforces our definitions, and useful because it reduces time spent searching and clarifying.

Enterprise-grade governance, access control, persistence, and integrations would be addressed prior to any production deployment.
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
- The knowledge base is built from **uploaded docs only** (not yet connected to Drive/Confluence/SharePoint)
- Citations are not shown as clickable sources yet (can be added next)
- Index is stored in memory per session (can be upgraded to persistent storage)

### Data handling note
Avoid uploading restricted or highly sensitive documents until we add governance controls:
access control, audit logs, encryption, and retention policies.
        """
    )

# =========================
# TAB 4: ARCHITECTURE (ASCII DIAGRAMS)
# =========================
with tab_arch:
    st.header("Architecture & Technical Details")

    st.markdown(
        """
Below are two diagrams describing how data flows through the system today (prototype) and how it would look in a production-ready enterprise deployment.
        """
    )

    st.subheader("Data Flow: Current Prototype (Ephemeral Only)")

    diagram_ephemeral = r"""
+------------------------+        +-------------------------+
| Marketing Team Member  |        |       Streamlit UI      |
| (upload + ask)         |------->|  upload + chat interface|
+------------------------+        +-----------+-------------+
                                            |
                                            | (session state)
                                            v
                                  +---------+----------+
                                  |  Session Memory    |
                                  | - uploaded_files   |
                                  | - chat_history     |
                                  | - rag_ready flag   |
                                  +---------+----------+
                                            |
                                            | (temp write for parsing)
                                            v
                                  +---------+----------+
                                  | Temp File (/tmp)   |
                                  | short-lived         |
                                  +---------+----------+
                                            |
                                            v
+---------------------+     +----------------+     +-------------------+
| Parse docs to text  |---->| Chunk text     |---->| Create embeddings |
+---------------------+     +----------------+     +---------+---------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | In-Memory Vector    |
                                                   | Store (ephemeral)   |
                                                   +----------+----------+
                                                              ^
                                                              |
                                        +---------------------+------------------+
                                        |     RAG Pipeline: Retrieve -> Answer   |
                                        +---------------------+------------------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | Answer shown in UI  |
                                                   +---------------------+

Notes:
- Temp files are deleted after parsing.
- Nothing is persisted across app restarts.
"""
    st.code(diagram_ephemeral, language="text")

    st.markdown(
        """
**What this means (prototype behavior):**
- Uploaded files are held in session memory and written briefly to a temporary filesystem for parsing.
- The app stores only embeddings and chunks in memory for retrieval during the session.
- When the app restarts or the session ends, the content is not retained (no persistence).
        """
    )

    st.divider()
    st.subheader("Data Flow: Production Evolution (Ephemeral + Persistent Layers)")

    diagram_production = r"""
+------------------------+      +---------------------+      +----------------------+
| Marketing Team Member  |----->| Internal Web UI     |----->| SSO / Identity (IdP) |
+------------------------+      +----------+----------+      +----------------------+
                                          |
                                          v
                                 +--------+--------+
                                 | Gateway / Proxy |
                                 | TLS + Rate Limit|
                                 +--------+--------+
                                          |
                                          v
                              +-----------+------------+
                              |  RAG Service            |
                              | Retrieve -> Reason ->   |
                              | Respond + Citations     |
                              +-----------+------------+
                                          |
                                          v
                             +------------+-------------+
                             | Access Control (RBAC/ACL)|
                             +------------+-------------+
                                          |
                                          v
                             +------------+-------------+
                             | Persistent Vector Store   |
                             | (Azure AI Search, etc.)   |
                             +------------+-------------+
                                          ^
                                          |
         +---------------------+----------+----------+----------------------+
         |                     |                     |                      |
         v                     v                     v                      v
+------------------+   +------------------+   +------------------+   +------------------+
| SharePoint /     |   | Confluence /     |   | MarTech Tools     |   | Data Platforms    |
| Drive Docs       |   | Wiki/Pages       |   | (AJO/CJA/SFMC)    |   | (Snowflake/DBX)   |
+--------+---------+   +--------+---------+   +--------+---------+   +--------+---------+
         \___________________________  Ingestion Service  ____________________________/
                                      (scheduled or on-demand)
                                              |
                                              v
                                 +------------+-------------+
                                 | Ephemeral Processing      |
                                 | parse -> chunk -> embed   |
                                 +------------+-------------+
                                              |
                                              v
                                 +------------+-------------+
                                 | Metadata Store            |
                                 | source, ACL, timestamps   |
                                 +------------+-------------+

Also:
- Audit logs capture who asked what, when, and what sources were used.
- Optional raw docs can be stored in controlled object storage (Blob/S3) if required.
"""
    st.code(diagram_production, language="text")

    st.markdown(
        """
**What this means (production-ready behavior):**
- Parsing/chunking/embedding still happens ephemerally, but the organization intentionally chooses what to persist.
- Persistent vector storage enables reliable, scalable retrieval across sessions.
- SSO + access control + audit logs enable enterprise governance and traceability.
- Integrations shift ingestion from manual uploads to managed knowledge pipelines.
        """
    )


# =========================
# Footer
# =========================
st.markdown(
    """
<hr style="margin-top:32px;margin-bottom:8px;">
<div style="text-align:center;font-size:12px;color:gray;">
  © 2025 Howard Nguyen, PhD. For prototype and demonstration only.
</div>
""",
    unsafe_allow_html=True,
)
