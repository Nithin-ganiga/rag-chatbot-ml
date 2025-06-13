import chromadb
from chromadb.utils import embedding_functions
import sys
import os
from typing import List, Dict, Any
import numpy as np
import time
import requests
import hashlib

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_DB_PATH, COLLECTION_NAME, HF_API_KEY, EMBEDDING_MODEL

# Import embedder functions
from embedding.embedder import get_embedding, compute_simple_embedding

# Create a custom HuggingFace embedding function for ChromaDB
class HuggingFaceEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.api_failures = 0
        self.max_api_failures = 3  # After this many failures, use local embeddings for the rest of the session
        self.use_local_only = False

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # If we've had too many API failures, switch to local-only mode
        if self.use_local_only:
            return [compute_simple_embedding(text) for text in texts]

        embeddings = []
        api_failed = False
        
        for text in texts:
            try:
                if not self.use_local_only:
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
                    
                    response = requests.post(url, headers=headers, json={"inputs": text})
                    
                    if response.status_code != 200:
                        print(f"Error in HuggingFace API: {response.text}")
                        # Local fallback
                        embedding = compute_simple_embedding(text)
                        api_failed = True
                    else:
                        embedding = response.json()
                else:
                    embedding = compute_simple_embedding(text)
                    
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding in vector store: {e}")
                embedding = compute_simple_embedding(text)
                embeddings.append(embedding)
                api_failed = True
        
        if api_failed:
            self.api_failures += 1
            if self.api_failures >= self.max_api_failures:
                print("Too many API failures, switching to local embedding generation for the rest of the session")
                self.use_local_only = True
                
        return embeddings

class VectorStore:
    def __init__(self):
        # Ensure the directory exists
        vector_db_path = str(VECTOR_DB_PATH)
        os.makedirs(vector_db_path, exist_ok=True)
        
        print(f"Initializing vector store with path: {vector_db_path}")
        
        # Create a client with the specified directory for persistence
        try:
            # Updated client configuration for latest ChromaDB
            self.client = chromadb.PersistentClient(
                path=vector_db_path
            )
            
            # Create HuggingFace embedding function for ChromaDB
            self.embedding_function = HuggingFaceEmbeddingFunction(
                api_key=HF_API_KEY or "",  # Ensure API key is never None
                model_name=EMBEDDING_MODEL
            )
            
            # Get or create the collection
            try:
                self.collection = self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function
                )
                print(f"Connected to collection: {COLLECTION_NAME}")
            except Exception as e:
                print(f"Error getting collection: {e}")
                raise
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store"""
        if not documents:
            print("No documents to add")
            return
            
        try:
            # Generate unique IDs
            timestamp = int(time.time())
            ids = [f"doc_{timestamp}_{doc.get('source', 'unknown').replace(' ', '_')}_{doc.get('chunk_id', i)}" for i, doc in enumerate(documents)]
            texts = [doc["text"] for doc in documents]
            metadatas = [{k: str(v) for k, v in doc.items() if k != "text" and k != "embedding"} for doc in documents]
            
            # Print some debug info
            print(f"Adding {len(documents)} documents to vector store")
            print(f"First document ID: {ids[0] if ids else 'none'}")
            print(f"First document text sample: {texts[0][:50] if texts else 'none'}...")
            
            # Add documents to the collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            print(f"Successfully added {len(documents)} documents to the vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def similar_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store"""
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            documents = []
            if results and 'documents' in results and results['documents'] and results['documents'][0]:
                for i, doc_text in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] else {}
                    document = {
                        "text": doc_text,
                        "id": results['ids'][0][i] if 'ids' in results and results['ids'] and results['ids'][0] else f"result_{i}",
                        **metadata
                    }
                    documents.append(document)
            
            return documents
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def delete_document_by_source(self, source_filename: str) -> bool:
        """Delete all documents from a specific source file"""
        try:
            # Get all document IDs with the matching source in metadata
            results = self.collection.get(
                where={"source": source_filename}
            )
            
            # If no documents found with this source
            if not results["ids"]:
                print(f"No documents found with source: {source_filename}")
                return False
                
            # Delete documents with matching IDs
            self.collection.delete(
                ids=results["ids"]
            )
            
            print(f"Deleted {len(results['ids'])} documents from source: {source_filename}")
            return True
        except Exception as e:
            print(f"Error deleting documents for source {source_filename}: {e}")
            return False
    
    def list_document_sources(self) -> List[str]:
        """Get a list of all unique document sources in the collection"""
        try:
            # This is a simple implementation that gets all documents
            # For larger collections, you would need pagination
            results = self.collection.get()
            
            # Extract unique sources from metadata
            sources = set()
            for metadata in results.get("metadatas", []):
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
                    
            return list(sources)
        except Exception as e:
            print(f"Error getting document sources: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def reset(self) -> None:
        """Delete the collection and recreate it"""
        try:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception as e:
                print(f"Error deleting collection (may not exist yet): {e}")
                
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            print(f"Reset collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Error resetting collection: {e}")
            raise
    
    def delete(self) -> None:
        """Delete the collection"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"Deleted collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
            raise
    
    def persist(self) -> None:
        """Persist the collection to disk"""
        # ChromaDB now handles persistence automatically
        pass 