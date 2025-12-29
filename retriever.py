"""Retriever module."""
import os
import tempfile
from typing import List, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_document
from llms import EMBEDDINGS

VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)


class DocumentRetriever(BaseRetriever):
    """A simple retriever that stores uploaded documents in an in-memory vector store."""

    k: int = 4

    def __init__(self, k: int = 4, **kwargs: Any):
        super().__init__(**kwargs)
        self.k = k
        self.documents: List[Document] = []

    def store_documents(self, docs: List[Document]) -> None:
        """Add docs to the vector store."""
        if not docs:
            return

        # Split to improve retrieval quality
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        VECTOR_STORE.add_documents(split_docs)

    def add_documents_from_uploads(self, uploaded_files: List[Any]) -> None:
        """Load Streamlit uploaded files and add to vector store."""
        docs: List[Document] = []

        for file in uploaded_files:
            # Save to temp file because many loaders want a file path
            suffix = os.path.splitext(file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getbuffer())
                temp_filepath = tmp.name

            try:
                file_docs = load_document(temp_filepath)
                docs.extend(file_docs)
            except Exception as e:
                print(f"Failed to load {file.name}: {e}")
            finally:
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass

        self.documents.extend(docs)
        self.store_documents(docs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementation for retriever."""
        if len(self.documents) == 0:
            return []
        return VECTOR_STORE.similarity_search(query=query, k=self.k)
