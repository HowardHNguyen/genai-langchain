"""LangGraph for RAG.

CorpDocs with Citations: A Corporate Documentation Pipeline with RAG and Source Attribution
This single file contains the complete code to run a documentation generation system
using LangChain, LangGraph, and Gradio. In addition to generating and refining documentation,
this pipeline now retrieves and attaches citations to the final output.

Workflow Overview:
1. Generate an initial project documentation draft from a user's request.
2. Analyze the draft for compliance with corporate standards.
3. If issues are detected, prompt for LLM feedback.
4. Finalize the documentation (integrating any feedback).
5. Retrieve and append citations to the final document.
6. Output the fully revised document with inline source citations.

Note: More details on performance measurement and observability will be covered in section 8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llms import chat_model
from retriever import DocumentRetriever

retriever = DocumentRetriever()


@dataclass
class State:
    """State for LangGraph pipeline."""

    messages: Annotated[List[BaseMessage], add_messages]
    docs: List[Document] | None = None
    answer: str | None = None


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful corporate documentation assistant. "
            "Use the provided context to answer the user and include citations when possible.",
        ),
        ("human", "{question}"),
        ("system", "Context:\n{context}"),
    ]
)


def retrieve(state: State) -> dict:
    """Retrieve relevant documents based on the latest user message."""
    question = state.messages[-1].content
    retrieved_docs = retriever.invoke(question)
    return {"docs": retrieved_docs}


def generate(state: State) -> dict:
    """Generate an answer using retrieved docs."""
    question = state.messages[-1].content
    docs = state.docs or []
    context = "\n\n".join([d.page_content for d in docs])

    chain = PROMPT | chat_model
    response = chain.invoke({"question": question, "context": context})
    return {"answer": response.content}


def double_check(state: State) -> dict:
    """Placeholder for validation/compliance checking logic."""
    # You can expand this later (policy checks, formatting, etc.).
    return {}


def doc_finalizer(state: State) -> dict:
    """Finalize and return the answer."""
    return {"messages": [AIMessage(state["answer"])]}


# Compile application and test
graph_builder = StateGraph(State).add_sequence(
    [retrieve, generate, double_check, doc_finalizer]
)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("doc_finalizer", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}
