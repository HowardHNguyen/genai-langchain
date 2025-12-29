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
"""LangGraph RAG pipeline (compatible: uses add_node + add_edge, no add_sequence)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llms import chat_model
from retriever import DocumentRetriever

# Shared retriever instance
retriever = DocumentRetriever()

# Graph runtime config (thread_id can be any stable value)
config = {"configurable": {"thread_id": "abc123"}}


@dataclass
class State:
    """State for LangGraph pipeline."""

    # Chat messages (LangGraph helper accumulates messages)
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)

    # Retrieved docs + final answer
    docs: Optional[List[Document]] = None
    answer: Optional[str] = None


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful corporate documentation assistant. "
            "Use the provided context to answer the user. "
            "If the context is insufficient, say what’s missing and ask a clarifying question.",
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
    """Generate answer using retrieved docs."""
    question = state.messages[-1].content
    docs = state.docs or []
    context = "\n\n".join([d.page_content for d in docs])

    chain = PROMPT | chat_model
    response = chain.invoke({"question": question, "context": context})
    return {"answer": response.content}


def finalize(state: State) -> dict:
    """Return the final answer as an AIMessage appended to messages."""
    answer = state.answer or "I couldn’t generate an answer. Please try again."
    return {"messages": [AIMessage(content=answer)]}


# ---- Build graph (compatible approach) ----
graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("finalize", finalize)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", "finalize")
graph_builder.add_edge("finalize", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
