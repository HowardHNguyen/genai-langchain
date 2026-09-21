# AI Knowledge Platform — LangGraph RAG prototype

A Streamlit knowledge assistant for anyone who wants to explore their documents—for work, study, research, or personal learning.
Upload documents, build a session-owned index, and ask questions with numbered source excerpts.
For an exact numbered overview question matching a named source table, the app extracts the complete list verbatim with source citations; it does not ask the model to reconstruct that table. Explanations, partial/conflicting tables, and broader questions use the generation pipeline.

The graph performs follow-up question rewriting, hybrid keyword/semantic retrieval, complete table/passage expansion, and grounded generation. It can retry once with previously unseen evidence after an insufficient-evidence response.
It is a fixed RAG workflow, not an autonomous tool-using agent or an approval system.

## Run locally

Use Python 3.12 (also select 3.12 in Streamlit Cloud).

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit the local secrets file with your own keys; do not commit it.
streamlit run streamlit_app.py
```

The dependency set is pinned, including transitive dependencies. No Pandoc, LibreOffice,
unstructured, external vector database, or filesystem embedding cache is needed.

## Streamlit Cloud deployment

1. Deploy this repository with entry point `streamlit_app.py` and Python 3.12.
2. In App settings → Secrets, retain your `OPENAI_API_KEY` and `GROQ_API_KEY`, and set:

   ```toml
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```

3. Reboot the app after updating code/secrets to dispose of the old process-wide index/checkpoint
   and load the new dependencies. If an old `cache/` directory remains in the deployment,
   remove it or redeploy onto a fresh filesystem; the new app never reads or creates it.
4. Click **Check Groq connection**. This checks model listing and a tiny synthetic generation request;
   it does not send uploaded documents, but can incur a small API charge.
5. Build a knowledge base with a harmless sample TXT file and verify an answer with a source excerpt.
6. Test a second browser session with a different document, then test Clear Session and follow-up questions.

### The previous `groq.NotFoundError`

This is a 404 response from Groq. It is not evidence of a disconnected API key.
The old code hard-coded `llama-3.3-70b-versatile`. At upgrade time (2026-09-20),
[Groq's model list](https://console.groq.com/docs/models) labels that model Enterprise,
while `openai/gpt-oss-20b` is listed as a standard production text model. Account availability
can differ. Verify the configured model in your own console and with the connection check.

The new default is `openai/gpt-oss-20b`, configurable with `GROQ_MODEL`.
There is no silent fallback to another provider/model. Groq calls use the official endpoint explicitly.
A rejected key (401), permission failure (403), missing resource/model (404), quota/rate limit (429),
server error, and timeout receive distinct sanitized guidance. See [Groq error codes](https://console.groq.com/docs/errors).
Do not paste real API keys into issues or chat. A 404's exact cause still requires account-level verification.

## Architecture and isolation

```text
Session uploads → structure-aware memory parsing → complete table/text parents
                → 1,400-character search chunks / 200-character overlap
                → OpenAI text-embedding-3-small (512 dimensions), 32-chunk batches
                → session-owned vector index + BM25 keyword index
Question + bounded history → optional Groq rewrite → hybrid search / heading ranking
                           → expand matching chunks to complete tables/passages
                           → token-bounded Groq generation → citation-ID validation
                           → formatted answer + original source excerpts
Insufficient evidence → one broader search when new passages are available
```

- No module-global retriever, vector store, chat model, or conversation checkpoint.
- Streamlit session state owns each index and conversation. The graph receives history explicitly.
- A changed successful build replaces the index and clears conversation. Failed builds preserve the previous
  index and history, but chat is disabled whenever the selected files do not match that index.
- Identical file content is indexed once. Rebuilding an unchanged selection does not call embeddings again.
- Clearing the session discards the index, uploaded widget state, and conversation, with a new widget identity.
- Recent successful conversation: at most six exchanges / 12,000 characters, further limited to 1,200 tokenizer tokens in each model request. UI: latest 20 turns.
- Model input is budgeted to 5,000 estimated tokens including instructions, question, history and evidence. Complete tables are prioritized over older conversation; a passage is skipped rather than silently truncated. Output is capped at 2,048 tokens, with a visible notice if the model reaches that cap. Token estimates reserve framing space; provider/account rate limits still apply.
- Overview/list retrieval starts with 10 hybrid-ranked parent passages plus same-table row groups and neighboring text, bounded to 32,000 characters before prompt packing. Broader retry searches up to 20 parents / 48,000 characters and prioritizes unseen passages.
- A session schema version resets legacy in-memory indexes after this upgrade. Rebuild uploaded documents once.

## Files and limits

PDF (text-based), UTF-8 TXT, DOCX (paragraphs and tables), and EPUB (spine order) are supported.
Convert legacy `.doc` to `.docx`; OCR scanned PDFs before upload. Password-protected PDFs are rejected.
Limits: 10 files; **100 MB/file**; **200 MB total**; **3,000 PDF pages/file**; 8 million extracted characters/file;
12 million extracted characters/selection; 200 MB expanded ZIP content; 10,000 archive entries;
12,000 searchable chunks/index; 4,000 characters/question. Limits are centralized in `limits.py`; the Streamlit server upload setting matches the 100 MB per-file limit.
Large documents index in batches with visible progress. Expanding the upload limit does not remove all memory, text-volume, or provider quota limits.
All selected files must parse before indexing commits. The app reports file-level parsing errors.
These limits reduce resource use; parsers are not a sandbox against every malicious document.

## Data handling and limitations

- Application code parses uploads in memory and creates no document or embedding disk cache.
- Document text and search queries are sent to **OpenAI** for embeddings. Questions, recent conversation,
  and retrieved passages are sent to **Groq** for rewriting/generation. Provider policies/settings govern their records.
- Clear Session drops application references; it is not secure memory erasure or provider-side deletion.
  Streamlit may retain disconnected sessions temporarily. Sessions are browser sessions, not authenticated identities.
- Application logs contain error class/provider/status only. Parser diagnostics that could include document bytes
  are suppressed. Do not enable HTTP debug logging or external prompt tracing for sensitive documents.
- Retrieved content is JSON-encoded user-role data, separate from system instructions. Prompts instruct the model
  to disregard document instructions. No tool execution is exposed. This mitigates, but cannot eliminate, prompt injection.
- Citation validation normalizes common reference formats and removes out-of-range markers. Missing or invalid references produce an explicit review warning alongside the answer and retrieved excerpts; they do not turn the whole answer into an opaque refusal. Answers with citation warnings are excluded from follow-up history. Excerpts are real retrieved passages;
  reference validation does not prove every claim is entailed by a passage. Review answers against sources.
- Model Markdown renders as formatted text and tables through a parser with raw HTML disabled. Active links and images are removed before Streamlit sanitizes the generated HTML. Source excerpts remain plain text.
- This public prototype has no SSO, tenant authorization, durable storage, global rate limits, or billing enforcement.
  The short per-session cooldown is a UI aid, not an abuse defense. Use non-sensitive sample documents and provider spend limits.

## Validation

```sh
pip check
python -m unittest discover -s tests -v
```

Tests use fake models/embeddings and synthetic documents; no API keys or paid calls are required.
Coverage includes cross-session isolation, atomic/batched rebuilds, deduplication, complete ten-row table retrieval,
501-page PDF parsing, an upload larger than 10 MB, input-token budgets, safe formatted rendering,
follow-up context, source references, provider errors, startup, and Streamlit session reset.
Regression fixtures are synthetic; private user documents are never committed.
GitHub Actions runs the same checks on Linux/Python 3.12.

For semantic quality and prompt-injection resistance, use the manual evaluation cases in
`EVALUATION.md` with the configured live model. Offline tests do not establish live model quality or account access.

## Roadmap

Add authentication and authorization, centralized usage controls, labeled retrieval/answer evaluations,
then tenant-scoped durable storage and explicit human-approval nodes before external actions.
`knowledge_base.json` is an unused historical sample and is not ingested automatically.

© 2026 Howard Nguyen, PhD. Prototype provided for demonstration and internal evaluation.
