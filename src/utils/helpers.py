import os
import sys
import time
import traceback
import shutil
import json
from typing import List, Dict, Any, BinaryIO
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from ingestion.pdf_loader import process_pdf, save_uploaded_pdf
from embedding.embedder import embed_documents
from retrieval.vector_store import VectorStore
from llm.chat import query_with_sources
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_PATH

# Add this to the imports section
import datetime

def process_uploaded_file(uploaded_file: BinaryIO) -> Dict[str, Any]:
    """Process an uploaded PDF file and add to vector store"""
    start_time = time.time()
    
    try:
        # Validate the uploaded file
        if uploaded_file is None:
            return {
                "success": False,
                "message": "No file was uploaded.",
                "time_taken": time.time() - start_time
            }
        
        # Save the uploaded file
        print(f"Processing uploaded file: {getattr(uploaded_file, 'name', 'unnamed')}")
        file_path = save_uploaded_pdf(uploaded_file)
        
        if not file_path or not os.path.exists(file_path):
            return {
                "success": False,
                "message": "Failed to save uploaded file.",
                "time_taken": time.time() - start_time
            }
            
        filename = os.path.basename(file_path)
        print(f"File saved as: {filename} at {file_path}")
        
        # Process the PDF
        chunks = process_pdf(file_path, filename)
        
        if not chunks:
            return {
                "success": False,
                "message": f"Failed to process {filename}. No text extracted.",
                "time_taken": time.time() - start_time
            }
        
        print(f"Extracted {len(chunks)} chunks from {filename}")
        print(f"Sample text from first chunk: {chunks[0]['text'][:100]}...")
        
        # Add to vector store
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        
        # After adding to the vector store, confirm documents are searchable
        check_result = check_document_searchable(filename)
        if check_result["success"]:
            success_message = f"Successfully processed {filename} with {len(chunks)} chunks."
            print(success_message)
            return {
                "success": True,
                "message": success_message,
                "document_count": vector_store.get_document_count(),
                "time_taken": time.time() - start_time
            }
        else:
            warning_message = f"Processed {filename}, but may have issues with retrieval: {check_result['message']}"
            print(warning_message)
            return {
                "success": True,
                "message": warning_message,
                "document_count": vector_store.get_document_count(),
                "time_taken": time.time() - start_time
            }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing uploaded file: {e}")
        print(error_trace)
        return {
            "success": False,
            "message": f"Error processing file: {str(e)}",
            "time_taken": time.time() - start_time
        }

def check_document_searchable(filename: str) -> Dict[str, Any]:
    """Check if a document is properly searchable in the vector store"""
    try:
        # Create a simple test query from the filename
        test_query = f"information about {os.path.splitext(filename)[0]}"
        
        # Try to retrieve documents based on the query
        vector_store = VectorStore()
        results = vector_store.similar_search(test_query, top_k=1)
        
        if results and len(results) > 0:
            return {
                "success": True,
                "message": f"Document '{filename}' is searchable"
            }
        else:
            return {
                "success": False,
                "message": f"Document '{filename}' was added but may not be retrievable. Check embedding service."
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error checking document searchability: {str(e)}"
        }

def diagnose_vector_store() -> Dict[str, Any]:
    """Run diagnostics on the vector store and return detailed information"""
    try:
        vector_store = VectorStore()
        
        # Get all document sources
        sources = vector_store.list_document_sources()
        count = vector_store.get_document_count()
        
        # Get some sample documents from each source
        samples = {}
        for source in sources:
            try:
                # Create a query based on the source name
                query = source.split(".")[0]  # Use filename without extension
                results = vector_store.similar_search(query, top_k=2)
                
                if results:
                    samples[source] = {
                        "count": len([r for r in results if r.get("source") == source]),
                        "sample_text": results[0]["text"][:100] + "..." if results else "No sample available"
                    }
                else:
                    samples[source] = {
                        "count": 0,
                        "sample_text": "No results found"
                    }
            except Exception as e:
                samples[source] = {
                    "error": str(e)
                }
        
        # Get database path info
        db_info = {
            "path": str(VECTOR_DB_PATH),
            "exists": os.path.exists(VECTOR_DB_PATH),
            "size_kb": round(sum(f.stat().st_size for f in Path(VECTOR_DB_PATH).glob('**/*') if f.is_file()) / 1024, 2) if os.path.exists(VECTOR_DB_PATH) else 0
        }
        
        return {
            "success": True,
            "document_count": count,
            "sources": sources,
            "samples": samples,
            "db_info": db_info,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error diagnosing vector store: {e}")
        print(error_trace)
        return {
            "success": False,
            "message": f"Error diagnosing vector store: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        }

def get_answer(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Get an answer to a query from the RAG system"""
    start_time = time.time()
    
    try:
        # Get relevant documents from vector store
        vector_store = VectorStore()
        relevant_docs = vector_store.similar_search(query, top_k=top_k)
        
        # Log query and retrieval for debugging
        print(f"Query: '{query}'")
        print(f"Retrieved {len(relevant_docs)} documents")
        for i, doc in enumerate(relevant_docs):
            source = doc.get("source", "Unknown")
            print(f"Document {i+1} from {source}: {doc['text'][:50]}...")
        
        if not relevant_docs:
            # Provide a general answer when no documents are found
            return {
                "answer": "I don't have any documents to reference for that question. Please upload relevant PDF documents so I can provide a more specific answer. In the meantime, I can try to help with general knowledge questions.",
                "sources": [],
                "time_taken": time.time() - start_time
            }
        
        # Get answer from LLM with context
        from llm.chat import query_with_sources  # Import here to avoid circular imports
        result = query_with_sources(query, relevant_docs)
        result["time_taken"] = time.time() - start_time
        
        return result
    except Exception as e:
        # Fallback to direct query if the vector store has issues
        print(f"Error querying vector store: {e}")
        try:
            # Fall back to direct query without context
            from llm.chat import query_with_sources
            result = query_with_sources(query, [])
            result["time_taken"] = time.time() - start_time
            return result
        except Exception as e2:
            print(f"Error with fallback query: {e2}")
            return {
                "answer": "I encountered an error while processing your question. Please try again later.",
                "sources": [],
                "time_taken": time.time() - start_time
            }

def reset_vector_store() -> Dict[str, Any]:
    """Reset the vector store and clean up data directories"""
    try:
        # Reset vector store
        vector_store = VectorStore()
        vector_store.reset()
        
        # Clean up raw and processed data
        cleanup_result = cleanup_data_directories()
        
        if cleanup_result["success"]:
            return {
                "success": True,
                "message": "Vector store reset and data directories cleaned successfully."
            }
        else:
            return {
                "success": True,
                "message": f"Vector store reset successfully, but data cleanup had issues: {cleanup_result['message']}"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to reset vector store: {e}"
        }

def get_vector_store_info() -> Dict[str, Any]:
    """Get information about the vector store"""
    try:
        vector_store = VectorStore()
        return {
            "success": True,
            "document_count": vector_store.get_document_count()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to get vector store info: {e}"
        }

def cleanup_data_directories() -> Dict[str, Any]:
    """Clean up raw and processed data directories while preserving vector store structure"""
    try:
        # Track which files are being deleted to clean their vector store entries
        deleted_files = []
        
        # Clean raw data directory
        raw_dir = RAW_DATA_DIR
        for filename in os.listdir(raw_dir):
            file_path = os.path.join(raw_dir, filename)
            if os.path.isfile(file_path) and not filename.startswith('.gitkeep'):
                try:
                    os.remove(file_path)
                    deleted_files.append(filename)
                    print(f"Deleted raw file: {file_path}")
                except PermissionError:
                    print(f"Permission error deleting {file_path} - file may be in use")
        
        # Reset vector store if we've deleted files
        if deleted_files:
            try:
                # We could delete individual entries, but a full reset is more reliable
                # when doing bulk cleanup
                vector_store = VectorStore()
                vector_store.reset()
                print("Reset vector store collection after file cleanup")
            except Exception as e:
                print(f"Error resetting vector store during cleanup: {e}")
        
        # Clean other items in processed data directory except the vector_store folder itself
        processed_dir = PROCESSED_DATA_DIR
        for item in os.listdir(processed_dir):
            item_path = os.path.join(processed_dir, item)
            if item != 'vector_store' and not item.startswith('.gitkeep'):
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        print(f"Deleted processed file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"Deleted processed directory: {item_path}")
                except PermissionError:
                    print(f"Permission error deleting {item_path} - file may be in use")
        
        return {
            "success": True,
            "message": "Data directories cleaned successfully"
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error cleaning data directories: {e}")
        print(error_trace)
        return {
            "success": False,
            "message": f"Error cleaning data directories: {str(e)}"
        }

def end_session_cleanup() -> Dict[str, Any]:
    """Cleanup function to be called when session ends"""
    try:
        # Reset vector store
        vector_store = VectorStore()
        vector_store.reset()
        
        # Clean data directories
        cleanup_result = cleanup_data_directories()
        
        return {
            "success": True,
            "message": "Session ended, data and vector store cleaned up"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to clean up session data: {e}"
        } 