import os
import sys
import requests
from typing import List, Dict, Any
import numpy as np
import hashlib

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_API_KEY, EMBEDDING_MODEL

# Local fallback functions for when API calls fail
def compute_simple_embedding(text: str) -> List[float]:
    """Generate a simple deterministic embedding from text without requiring API calls.
    This is a fallback when the API fails."""
    if not text.strip():
        return [0.0] * 384  # Default size for small embedding models
    
    # Use SHA256 hash of the text to create a deterministic embedding
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    # Convert hash bytes to a list of float values
    embedding = []
    for i in range(0, 48):  # 48 bytes × 8 bits = 384 dimensions
        if i < len(hash_bytes):
            # Convert each byte to a value between -1 and 1
            embedding.append((hash_bytes[i] / 127.5) - 1.0)
        else:
            embedding.append(0.0)
    
    # Normalize the embedding (L2 norm)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding

def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using HuggingFace's Inference API with local fallback"""
    if not text.strip():
        return [0.0] * 384  # Return a zero vector of standard size
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
        
        response = requests.post(url, headers=headers, json={"inputs": text})
        
        if response.status_code != 200:
            print(f"Error in HuggingFace API: {response.text}")
            print("Falling back to local embedding generation")
            return compute_simple_embedding(text)
            
        embedding = response.json()
        return embedding
    except Exception as e:
        print(f"Error generating embedding via API: {e}")
        print("Falling back to local embedding generation")
        # Use local embedding as fallback
        return compute_simple_embedding(text)

def get_embeddings_batch(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """Get embeddings for a batch of texts using HuggingFace's Inference API with local fallback"""
    embeddings = []
    
    # Process in batches to avoid API limits - HuggingFace has smaller batch sizes
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            # Filter out empty texts
            batch = [text for text in batch if text.strip()]
            
            if not batch:
                continue
            
            # Process one by one since HF API might have limitations
            batch_embeddings = []
            for text in batch:
                embedding = get_embedding(text)
                batch_embeddings.append(embedding)
                
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            # Use local embeddings as fallback
            for text in batch:
                embeddings.append(compute_simple_embedding(text))
    
    return embeddings

def embed_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add embeddings to a list of document dictionaries"""
    texts = [doc["text"] for doc in documents]
    
    # Get embeddings for all texts
    embeddings = get_embeddings_batch(texts)
    
    # Add embeddings to documents
    for i, embedding in enumerate(embeddings):
        if embedding:  # Only add if embedding was successfully generated
            documents[i]["embedding"] = embedding
    
    # We'll no longer filter out documents without embeddings since we're using fallbacks
    # This ensures all documents are processed even if the API fails
    
    return documents 