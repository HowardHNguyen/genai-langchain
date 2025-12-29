"""Retriever module (Pydantic-safe for LangChain BaseRetriever)."""

import os
import tempfile
from typing import List, Any

from pydantic.v1 import Field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_document
from llms import EMBEDDINGS

# One in-memory vector store for the app session
VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)


class DocumentRetriever(BaseRetriever):
    """Stores documents in an in-memory vector store and retrieves by similarity search."""

    k: int = 4
    documents: List[Document] = Field(default_factory=list)

    def store_documents(self, docs: List[Document]) -> None:
        """Split and add docs to the vector store."""
        if not docs:
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        VECTOR_STORE.add_documents(split_docs)

    def add_documents_from_uploads(self, uploaded_files: List[Any]) -> None:
        """Load Streamlit uploaded files and add them to the vector store."""
        docs: List[Document] = []

        for file in uploaded_files:
            suffix = os.path.splitext(file.name)[-1]

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getbuffer())
                temp_filepath = tmp.name

            try:
                file_docs = load_document(temp_filepath)
                docs.extend(file_docs)
            except Exception as e:
                # Keep app running even if one doc fails
                print(f"Failed to load {file.name}: {e}")
            finally:
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass

        # Update stored docs and vector store
        if docs:
            self.documents.extend(docs)
            self.store_documents(docs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant chunks from the vector store."""
        if not self.documents:
            return []
        return VECTOR_STORE.similarity_search(query=query, k=self.k)
