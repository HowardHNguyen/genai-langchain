"""Run with: streamlit run streamlit_app.py"""
import logging
import time
from uuid import uuid4
import streamlit as st
from config import Settings, ConfigurationError
from document_loader import SUPPORTED_EXTENSIONS
from llms import create_chat_model, create_embeddings, check_groq, ServiceError
from retriever import DocumentRetriever, Upload, BuildError, selection_signature
from rag import ask
from answer_rendering import answer_html, ANSWER_CSS
from limits import MAX_FILE_MB, MAX_BATCH_MB, MAX_PDF_PAGES, MAX_FILES, MAX_TEXT_CHARS, MAX_INDEX_CHARS, MAX_CHUNKS

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="AI Knowledge Platform", page_icon="📚", layout="wide")


def reset_session():
    """Drop references to all documents, embeddings, messages, and uploaded bytes."""
    for key in list(st.session_state):
        if key.startswith("kb_") or key.startswith("upload_"):
            del st.session_state[key]
    st.session_state.kb_id = uuid4().hex
    st.session_state.kb_history = []
    st.session_state.kb_turns = []
    st.session_state.kb_retriever = None
    st.session_state.kb_schema = 2


if "kb_id" not in st.session_state or st.session_state.get("kb_schema") != 2:
    reset_session()
settings = Settings.load()
st.title("📚 AI Knowledge Platform")
st.caption("Document-grounded answers with source references · LangGraph RAG prototype")

with st.sidebar:
    st.subheader("Model connection")
    st.caption("Groq generation model")
    st.code(settings.model, language=None)
    if st.button("Check Groq connection"):
        now = time.monotonic()
        if now - st.session_state.get("kb_last_check", 0) < 10:
            st.info("Please wait 10 seconds before checking again.")
        else:
            st.session_state.kb_last_check = now
            try:
                with st.spinner("Checking model access..."):
                    st.success(check_groq(settings))
            except (ConfigurationError, ServiceError) as exc:
                st.error(str(exc))
            except Exception as exc:
                LOGGER.error("Connection check failed type=%s", type(exc).__name__)
                st.error("The connection check could not finish. The owner should check the deployment configuration.")
    st.caption("This test sends only a short synthetic prompt to Groq and may incur a small API charge.")
    st.button("Clear Session", on_click=reset_session, type="secondary")
    st.caption("Clears this session's index, uploaded files, and conversation. Provider records are outside this control.")

chat, about, howto, architecture = st.tabs(["💬 Chat", "ℹ️ About", "🧭 How to use", "🏗️ Architecture"])
with chat:
    st.subheader("Your documents")
    st.caption("Build sends extracted text to OpenAI for embeddings. Questions, recent conversation, and selected passages are sent to Groq. Only upload content you are authorized to share with these providers.")
    uploaded = st.file_uploader("Upload documents", type=list(SUPPORTED_EXTENSIONS),
                                accept_multiple_files=True, key=f"upload_{st.session_state.kb_id}",
                                max_upload_size=MAX_FILE_MB)
    st.caption(f"Up to {MAX_FILE_MB} MB per file · {MAX_BATCH_MB} MB total · {MAX_PDF_PAGES:,} PDF pages per file")
    uploads = [Upload(file.name, file.getvalue()) for file in (uploaded or [])]
    signature = selection_signature(uploads)
    retriever = st.session_state.kb_retriever
    active = retriever is not None and retriever.ready and retriever.signature == signature
    if uploads and not active:
        st.info("Build the knowledge base for this selection before asking a question.")
    if st.button("Build Knowledge Base", disabled=not uploads, type="primary"):
        try:
            with st.spinner("Parsing and indexing documents..."):
                if retriever is None:
                    retriever = DocumentRetriever(create_embeddings(settings))
                indicator = st.progress(0, text="Preparing documents…")
                try:
                    changed = retriever.build(uploads, progress=lambda fraction, message: indicator.progress(fraction, text=message))
                finally:
                    indicator.empty()
                st.session_state.kb_retriever = retriever
                active = True
                if changed:
                    st.session_state.kb_history = []
                    st.session_state.kb_turns = []
            st.success(f"Ready: {retriever.file_count} unique files, {retriever.chunk_count} searchable chunks." if changed else "This selection is already indexed; no duplicate chunks were added.")
        except BuildError as exc:
            st.error(str(exc))
            for name, problem in exc.errors:
                st.text(f"{name}: {problem}")
        except (ConfigurationError, ServiceError) as exc:
            st.error(str(exc))
        except Exception as exc:
            LOGGER.error("Indexing failed type=%s", type(exc).__name__)
            st.error("Indexing could not finish. The existing knowledge base was not changed. Try smaller files or contact the owner.")
    if active:
        st.caption(f"Knowledge base ready · {retriever.file_count} unique files · {retriever.chunk_count} chunks")
    st.divider()

    def show_turn(turn):
        with st.chat_message("user"):
            st.text(turn["question"])
        with st.chat_message("assistant"):
            st.html(ANSWER_CSS + answer_html(turn["answer"]))
            for source in turn.get("sources", []):
                location = f" · page {source['page']}" if source.get("page") else ""
                if source.get("section"):
                    location += f" · section {source['section']}"
                with st.expander(f"[{source['id']}] Source passage"):
                    st.text(source["source"] + location)
                    if source.get("section_title"):
                        st.text(source["section_title"])
                    st.text(source["text"])

    for turn in st.session_state.kb_turns:
        show_turn(turn)
    question = st.chat_input("Ask about your selected documents...", disabled=not active, max_chars=4000)
    if question:
        now = time.monotonic()
        if now - st.session_state.get("kb_last_question", 0) < 3:
            st.info("Please wait a few seconds before asking again.")
        else:
            st.session_state.kb_last_question = now
            try:
                with st.spinner("Finding evidence and preparing an answer..."):
                    result = ask(retriever, create_chat_model(settings), question, st.session_state.kb_history)
                turn = {"question": question, "answer": result["answer"], "sources": result["sources"]}
                st.session_state.kb_turns = (st.session_state.kb_turns + [turn])[-20:]
                # Do not reuse unsuccessful/ungrounded responses as conversation context.
                if result["sources"]:
                    st.session_state.kb_history = (st.session_state.kb_history + [(question, result["answer"])])[-6:]
                show_turn(turn)
            except (ConfigurationError, ServiceError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                LOGGER.error("Answer failed type=%s", type(exc).__name__)
                st.error("The answer could not be completed. Please retry; contact the owner if this persists.")

with about:
    st.markdown("""
### Knowledge assistance for Marketing Operations and beyond
Upload guidelines, standards, playbooks, or operating models and ask questions grounded in their text.
The app retrieves passages, generates a cited answer, and shows the supporting excerpts.
It can help draft summaries and checklists for human review.

This is a **LangGraph-orchestrated RAG prototype**. Its workflow is fixed; it does not autonomously
choose tools, execute actions, or enforce business approvals. Those are future capabilities.
Source references make answers reviewable, but do not guarantee that every claim is correct.

### Data handling
- Files are parsed in memory; the application writes no document or embedding cache to disk.
- Each Streamlit session owns a separate index and conversation. No shared checkpoint is used.
- Clear Session drops the app's references to that session's uploads, index, and conversation.
  This is not a secure memory wipe and cannot erase provider-side records.
- OpenAI processes document text and search queries for embeddings. Groq processes questions,
  bounded recent conversation, and retrieved text for generation.
- Provider retention and account settings apply independently. Do not assume zero retention.
- Application error logs contain error types/status codes, not prompts or document contents.

This public demo has no enterprise authentication, durable storage, or organization-wide usage controls.
Use non-sensitive demonstration documents. Enterprise deployment requires an access and data-governance review.
""")
with howto:
    st.markdown(f"""
1. Check the Groq connection in the sidebar if generation is failing.
2. Select PDF, UTF-8 TXT, DOCX, or EPUB files. Convert legacy DOC files to DOCX; run OCR on scanned PDFs first.
3. Click **Build Knowledge Base**. All selected files must parse successfully. Rebuilding the same selection is a no-op.
4. Ask a specific question, then inspect the numbered source passages. Follow-up questions use recent successful conversation.
5. Removing or replacing files disables chat until you rebuild. A successful changed build starts a new conversation.
6. Use **Clear Session** to remove the index, files, and conversation for this browser session.

Limits: {MAX_FILES} files, {MAX_FILE_MB} MB per file, {MAX_BATCH_MB} MB total, {MAX_PDF_PAGES:,} PDF pages per file,
{MAX_TEXT_CHARS:,} extracted characters per file, {MAX_INDEX_CHARS:,} across the selection, and {MAX_CHUNKS:,} searchable chunks per knowledge base. Questions are limited to 4,000 characters.
The model receives at most six recent successful exchanges (12,000 characters), plus complete retrieved passages within a bounded context budget.
Tables retain their headings and rows. Keyword and semantic search are combined; an insufficient-evidence response triggers one broader search.
Large documents are indexed in batches with progress updates and can take several minutes.
The UI retains the latest 20 turns. Unsupported, corrupted, or oversized files produce explicit errors.

If Groq reports **not found**, ask the owner to check `GROQ_MODEL` against the account's current model list.
Invalid keys, access restrictions, quota limits, and timeouts have separate messages.
""")
with architecture:
    st.code("""Browser session
  ├─ Selected files → in-memory parsing → chunking → OpenAI embeddings
  │                                              → session-owned vector index
  └─ Question + bounded conversation → standalone search query (Groq)
                                      → hybrid keyword + semantic search
                                      → expand to complete tables and passages
                                      → grounded answer (Groq; one broader retry if needed)
                                      → source-ID validation → formatted answer
                                      → answer + source excerpts

Clear Session → discard session-owned index, uploads, conversation, and widget state
No global retriever · no global checkpoint · no disk embedding cache""", language=None)
    st.markdown("""
Retrieved text is supplied as untrusted data in a user-role message, separate from system instructions.
The graph exposes no action tools. Model Markdown is rendered with raw HTML, active links, and remote images disabled. Numeric citation IDs are checked against retrieved passages;
this validates reference existence, not factual entailment. Prompt injection remains a model-level risk.

Next steps toward an enterprise platform: authentication, per-user authorization, durable tenant-scoped storage,
central cost controls, evaluation against labeled examples, and explicit approval nodes for external actions.
""")
st.caption("© 2026 Howard Nguyen, PhD · Prototype and demonstration")
