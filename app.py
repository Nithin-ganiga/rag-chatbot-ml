import streamlit as st
import os
import sys
from datetime import datetime
import atexit
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import local modules
from src.utils.helpers import (
    process_uploaded_file, get_answer, reset_vector_store, get_vector_store_info,
    end_session_cleanup, cleanup_data_directories, diagnose_vector_store, check_document_searchable
)
from src.config import APP_TITLE, APP_DESCRIPTION, RAW_DATA_DIR
from src.retrieval.vector_store import VectorStore

# Check for essential env variables
if not os.environ.get("RAW_DATA_DIR"):
    print("WARNING: RAW_DATA_DIR not set in environment. Using default path.")
    os.environ["RAW_DATA_DIR"] = str(Path(os.path.dirname(__file__)) / "data/raw")

# Register cleanup function
atexit.register(end_session_cleanup)

# App config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling for fixed chat input and better layout
st.markdown("""
    <style>
    /* Chat input fixed and properly spaced from sidebar */
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 1rem;
        left: 22rem; /* shift right to account for sidebar */
        width: calc(100% - 24rem); /* full width minus sidebar and padding */
        z-index: 999;
        background-color: #0e1117;
        padding-right: 2rem;
    }

    /* Add padding to bottom of chat messages container so last message isn't hidden */
    .element-container:has([data-testid="stChatMessage"]) {
        padding-bottom: 6rem;
    }

    /* Fix sidebar width consistency */
    section[data-testid="stSidebar"] {
        width: 21rem;
    }

    /* Remove horizontal scroll */
    .main {
        overflow-x: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Session state init
for key, value in {"messages": [], "document_count": 0, "is_session_ending": False}.items():
    st.session_state.setdefault(key, value)

# Sidebar
with st.sidebar:
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    st.subheader("📊 Document Stats")
    try:
        info = get_vector_store_info()
        st.session_state.document_count = info.get("document_count", 0)
    except Exception as e:
        st.error(f"Error getting document info: {e}")
    st.metric("Documents in Database", st.session_state.document_count)

    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    if uploaded_file:
        with st.spinner('Processing document...'):
            result = process_uploaded_file(uploaded_file)
            msg = result.get("message", "Failed to process document")
            if result.get("success"):
                st.success(msg)
                st.session_state.document_count = result.get("document_count", 0)
            else:
                st.error(msg)

    st.subheader("🗑️ Reset Database")
    if st.button("Reset Document Database"):
        with st.spinner('Resetting database...'):
            result = reset_vector_store()
            if result.get("success"):
                st.success("Database reset successfully")
                st.session_state.document_count = 0
            else:
                st.error(result.get("message", "Failed to reset database"))

# Tabs for UI
chat_tab, file_tab, diag_tab = st.tabs(["Chat", "File Index", "Diagnostics"])

# Chat Tab
with chat_tab:
    st.markdown("""<h1 style='margin-bottom:0;'>💬 Hii how can I help you?</h1>""", unsafe_allow_html=True)

    # Handle new input
    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        # Add user message immediately
        user_msg = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%I:%M %p")
        }
        st.session_state.messages.append(user_msg)

        # Get answer
        with st.spinner("Thinking..."):
            result = get_answer(prompt)
            answer = result.get("answer", "I couldn't find an answer.")
            sources = [
                src["source"] if isinstance(src, dict) and "source" in src else src
                for src in result.get("sources", [])
            ]
            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%I:%M %p")
            }
            st.session_state.messages.append(assistant_msg)

    # Show chat only if messages exist
    if st.session_state.messages:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    for source in msg["sources"]:
                        st.info(f"Source: {source}")
                st.caption(msg["timestamp"])

# File Tab
with file_tab:
    st.title("📑 Uploaded Files")

    def get_uploaded_files():
        files = []
        if os.path.exists(RAW_DATA_DIR):
            for filename in os.listdir(RAW_DATA_DIR):
                path = os.path.join(RAW_DATA_DIR, filename)
                if os.path.isfile(path) and not filename.startswith('.gitkeep'):
                    files.append({
                        "name": filename,
                        "path": path,
                        "size": os.path.getsize(path) / 1024,
                        "date": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                    })
        return files

    def delete_file(path, filename):
        try:
            vector_store = VectorStore()
            vector_store.delete_document_by_source(filename)
            os.remove(path)
            return True
        except Exception as e:
            print(f"Error deleting file and vector entries: {e}")
            return False

    try:
        vector_store = VectorStore()
        vector_sources = vector_store.list_document_sources()
    except Exception as e:
        vector_sources = []
        print(f"Error listing document sources: {e}")

    files = get_uploaded_files()
    
    st.subheader("Vector Store Status")
    if vector_sources:
        st.write(f"Documents in vector store: {', '.join(vector_sources)}")
    else:
        st.write("No documents in vector store.")

    st.subheader("Uploaded Files")
    if not files:
        st.info("No files have been uploaded yet.")
    else:
        st.dataframe([
            {"File Name": f["name"], "Size (KB)": f"{f['size']:.2f}", "Upload Date": f["date"]}
            for f in files
        ])

        st.subheader("Delete Individual Files")
        col1, col2 = st.columns(2)
        with col1:
            selected = st.selectbox("Select file to delete:", [f["name"] for f in files])
        with col2:
            if st.button("Delete Selected File"):
                path = next((f["path"] for f in files if f["name"] == selected), None)
                if path and delete_file(path, selected):
                    st.success(f"Deleted {selected} and cleared related vector store entries")
                    st.rerun()
                else:
                    st.error(f"Failed to delete {selected}")

        st.subheader("Delete All Files")
        if st.button("Delete All Uploaded Files"):
            with st.spinner("Deleting all files..."):
                vector_store = VectorStore()
                vector_store.reset()
                result = cleanup_data_directories()
                if result.get("success"):
                    st.success("All files deleted and vector store cleared successfully")
                    st.rerun()
                else:
                    st.error(f"Error deleting files: {result['message']}")

# Diagnostics Tab
with diag_tab:
    st.title("🔍 System Diagnostics")
    
    st.write("This tab helps diagnose issues with document indexing and retrieval.")
    
    if st.button("Run Vector Store Diagnostics"):
        with st.spinner("Analyzing vector store..."):
            result = diagnose_vector_store()
            
            if result.get("success"):
                st.success(f"Diagnostics completed successfully")
                
                # Display general info
                st.subheader("General Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Document Count", result.get("document_count", 0))
                with col2:
                    st.metric("Source Count", len(result.get("sources", [])))
                
                # Display DB info
                st.subheader("Database Information")
                db_info = result.get("db_info", {})
                st.write(f"Database Path: {db_info.get('path', 'Unknown')}")
                st.write(f"Database Exists: {db_info.get('exists', False)}")
                st.write(f"Database Size: {db_info.get('size_kb', 0)} KB")
                
                # Display sources
                st.subheader("Document Sources")
                if result.get("sources"):
                    for source in result.get("sources", []):
                        with st.expander(f"Source: {source}"):
                            sample = result.get("samples", {}).get(source, {})
                            st.write(f"Documents found: {sample.get('count', 0)}")
                            
                            if sample.get("error"):
                                st.error(f"Error: {sample.get('error')}")
                            elif sample.get("sample_text"):
                                st.write("Sample text:")
                                st.code(sample.get("sample_text", "No text available"))
                else:
                    st.info("No document sources found in vector store.")
                
                # Test query form
                st.subheader("Test Query")
                test_query = st.text_input("Enter a test query to check document retrieval:")
                if test_query:
                    if st.button("Run Test Query"):
                        with st.spinner("Searching..."):
                            try:
                                vector_store = VectorStore()
                                results = vector_store.similar_search(test_query, top_k=3)
                                
                                if results:
                                    st.success(f"Query returned {len(results)} results")
                                    for i, doc in enumerate(results):
                                        source = doc.get("source", "Unknown")
                                        with st.expander(f"Result {i+1} from {source}"):
                                            st.write(doc.get("text", "No text"))
                                else:
                                    st.warning("No results found for this query.")
                            except Exception as e:
                                st.error(f"Error running query: {str(e)}")
            else:
                st.error(f"Error running diagnostics: {result.get('message', 'Unknown error')}")

    # Display API configuration info
    st.subheader("API Configuration")
    
    # Check API keys (without showing them)
    col1, col2 = st.columns(2)
    with col1:
        if os.environ.get("GROQ_API_KEY"):
            groq_status = "✅ Set" 
        else:
            groq_status = "❌ Not Set"
        st.write(f"GROQ API Key: {groq_status}")
    
    with col2:
        if os.environ.get("HF_API_KEY"):
            hf_status = "✅ Set"
        else:
            hf_status = "❌ Not Set"
        st.write(f"HuggingFace API Key: {hf_status}")
    
    st.write(f"Embedding Model: {os.environ.get('EMBEDDING_MODEL', 'Not Set')}")
    st.write(f"LLM Model: {os.environ.get('LLM_MODEL', 'Not Set')}")
    
    # Local paths check
    st.subheader("Local Paths")
    st.write(f"RAW_DATA_DIR: {RAW_DATA_DIR} (Exists: {os.path.exists(RAW_DATA_DIR)})")
    st.write(f"Working Directory: {os.getcwd()}")

# Footer
st.markdown("---")
st.markdown("🚀 Built by **Nithin Ganiga** using Streamlit, LangChain & Free APIs", unsafe_allow_html=True)

# End session
if st.sidebar.button("End Session and Clean Up"):
    with st.spinner("Cleaning up session data..."):
        result = end_session_cleanup()
        if result.get("success"):
            st.success("Session ended and data cleaned up successfully")
            st.session_state.update({"document_count": 0, "messages": []})
        else:
            st.error(f"Error during cleanup: {result['message']}")
