System Architecture

# AI Knowledge Platform

A document-grounded question-answering app: upload PDFs, DOCX, TXT or EPUB, build a private per-session knowledge base, and ask questions that are answered only from your own text, with numbered source references. The answer path is a LangGraph graph with a retrieval retry loop.

Streamlit UI LangGraph orchestration Hybrid retrieval (BM25 + vectors) OpenAI embeddings Groq generation Session-isolated · in-memory

## 01Ingestion & indexing

When you click **Build Knowledge Base**, files are parsed entirely in memory, split into passages, embedded, and committed to a session-owned index. Nothing is written to disk, and the index is rebuilt atomically — a failed build leaves the previous one untouched.

Parent passages keep tables and sections whole for context; smaller child chunks are what actually get embedded and searched. Both the dense vectors and the BM25 keyword index are built from the same chunks and live only in this browser session.

## 02The LangGraph answer graph

Each question runs through a compiled LangGraph state machine. The model never answers from general knowledge — it is given retrieved passages as untrusted data and must cite them. If the first pass finds no usable evidence, the graph widens the search once and retries before giving up.

main flow retry loop OpenAI call · Groq call in node label

The **rewrite** node turns a follow-up like “what about the second one?” into a standalone search query using recent conversation. **generate** validates every citation ID against the passages actually retrieved; an answer with unmatched or missing references is flagged rather than trusted. The dashed retry loop fires at most once.

## 03Components

Each module has a single responsibility, and the design keeps secrets, documents, and embeddings out of global state and off disk.

### streamlit_app.py

UI & session

Tabs for Chat, About, How-to, and Architecture. Owns per-session state (index, conversation, uploads), rate-limits questions, and renders answers with expandable source passages.

### document_loader.py

Parsing

In-memory extraction for PDF, TXT, DOCX, and EPUB. Preserves headings and table structure, enforces size and page limits, and refuses encrypted or scanned-only files.

### retriever.py

Hybrid search

BM25 keyword index plus a dense vector store, fused with reciprocal-rank fusion. Expands hits to whole parent passages and complete tables within a context budget; builds the index atomically.

### rag.py

LangGraph graph

Defines the rewrite → retrieve → generate → expand state machine, bounds prompt tokens, enforces the grounded system prompt, and normalizes and verifies citations.

### llms.py

Providers

Groq chat model (generation and query rewriting) and OpenAI embeddings, with lazy, bounded calls and sanitized provider errors that never leak keys, prompts, or document text.

### config.py · limits.py

Settings & guards

Reads keys from environment or Streamlit secrets without copying them into global state, and centralizes every size, page, chunk, and context limit.

## 04Safety & guardrails

- **Grounded-only answers.** The model may answer only from supplied passages, or return a fixed “not enough evidence” reply.
- **Prompt-injection containment.** Passages and filenames are passed as untrusted user-role data, never as instructions.
- **Citation validation.** Every inline reference ID is checked against retrieved passages; invalid ones are stripped and flagged.
- **No action tools.** The graph exposes no tools that act; it retrieves and answers only.
- **Rendered safely.** Model Markdown renders with raw HTML, active links, and remote images disabled.
- **Session isolation.** Each session has its own index and conversation; nothing is cached to disk or shared.

## 05Stack & what’s next

- **Framework.** Streamlit · LangGraph · LangChain core
- **Generation.** Groq (openai/gpt-oss-20b), temperature 0
- **Embeddings.** OpenAI text-embedding-3-small, 512-d
- **Retrieval.** In-memory vector store + BM25, RRF fusion
- **Quality.** CI tests for app, URLs, and retrieval quality
- **Toward enterprise.** auth, durable tenant storage, cost controls, labeled-set evaluation, approval nodes

© 2026 Howard Nguyen, PhD · LangGraph-orchestrated RAG prototype · architecture reference