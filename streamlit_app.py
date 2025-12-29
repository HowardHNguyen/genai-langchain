"""Streamlit app

Run this as follows:
> streamlit run streamlit_app.py
"""
import streamlit as st

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from langchain.schema import HumanMessage

from document_loader import DocumentLoader
from rag import graph, config, retriever

# Set page configuration
st.set_page_config(page_title="Corporate Documentation Manager", layout="wide")

# Initialize session state for chat history and file management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# Title and layout
st.title("📚 Corporate Documentation Manager")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat Interface")

    # Display chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # If RAG not ready, warn user
        if not st.session_state.rag_ready:
            assistant_response = (
                "Please upload documents first, then click **Build Knowledge Base**."
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        else:
            # Run LangGraph RAG pipeline
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                    )

                    # LangGraph returns messages; get last AI message content
                    final_message = result["messages"][-1].content
                    st.markdown(final_message)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": final_message}
            )

with col2:
    st.subheader("Document Management")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=list(DocumentLoader.supported_extensions),
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            existing_names = {f.name for f in st.session_state.uploaded_files}
            if file.name not in existing_names:
                st.session_state.uploaded_files.append(file)

    st.write("### Uploaded Files")
    if st.session_state.uploaded_files:
        for f in st.session_state.uploaded_files:
            st.write(f"- {f.name}")
    else:
        st.info("No files uploaded yet.")

    # Build KB button
    if st.button("Build Knowledge Base"):
        if not st.session_state.uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Indexing documents..."):
                # Add docs to retriever (vector store)
                retriever.add_documents_from_uploads(st.session_state.uploaded_files)
                st.session_state.rag_ready = True
            st.success("Knowledge base is ready! You can now ask questions.")

    # Clear state button
    if st.button("Clear Session"):
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        st.session_state.rag_ready = False
        st.success("Session cleared.")
