import os
"""Loading LLMs and Embeddings."""
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from config import set_environment

set_environment()

# Ensure local cache directory exists (Streamlit Cloud uses an ephemeral FS)
os.makedirs("./cache", exist_ok=True)

chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

store = LocalFileStore("./cache/")

underlying_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
# Avoiding unnecessary costs by caching the embeddings.
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
