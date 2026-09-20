"""Transactional, session-owned indexing with content-based deduplication."""
from dataclasses import dataclass
from hashlib import sha256
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_document, DocumentLoaderException
from llms import provider_call

MAX_FILES = 10
MAX_BATCH_BYTES = 30 * 1024 * 1024
MAX_CHUNKS = 1000


@dataclass(frozen=True)
class Upload:
    name: str
    data: bytes


def selection_signature(uploads):
    return tuple(sorted((u.name, sha256(u.data).hexdigest()) for u in uploads))


class BuildError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("Knowledge base was not changed. Fix the listed files and rebuild.")


class DocumentRetriever:
    def __init__(self, embeddings, k=4):
        self.embeddings = embeddings
        self.k = k
        self.store = None
        self.signature = None
        self.chunk_count = 0
        self.file_count = 0

    @property
    def ready(self):
        return self.store is not None and self.chunk_count > 0

    def build(self, uploads):
        if not uploads or len(uploads) > MAX_FILES:
            raise BuildError([("Selection", "Choose between 1 and 10 files.")])
        if sum(len(u.data) for u in uploads) > MAX_BATCH_BYTES:
            raise BuildError([("Selection", "Total upload size must not exceed 30 MB.")])
        signature = selection_signature(uploads)
        if signature == self.signature and self.ready:
            return False
        docs, errors, seen = [], [], set()
        for upload in uploads:
            digest = sha256(upload.data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                file_docs = load_document(upload.name, upload.data)
                for doc in file_docs:
                    doc.metadata["content_hash"] = digest
                docs.extend(file_docs)
            except DocumentLoaderException as exc:
                errors.append((upload.name, str(exc)))
        if errors:
            raise BuildError(errors)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        if not chunks or len(chunks) > MAX_CHUNKS:
            raise BuildError([("Selection", "Choose fewer or smaller documents (1–1,000 chunks required).")])
        # Build off to the side; failed embedding calls never mutate the active index.
        candidate = InMemoryVectorStore(self.embeddings)
        provider_call("OpenAI", candidate.add_documents, chunks)
        self.store = candidate
        self.signature = signature
        self.chunk_count = len(chunks)
        self.file_count = len(seen)
        return True

    def invoke(self, query):
        if not self.ready:
            return []
        return provider_call("OpenAI", self.store.similarity_search, query, k=self.k)
