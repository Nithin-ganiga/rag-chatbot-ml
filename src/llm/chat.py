import requests
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, LLM_MODEL

def create_prompt_with_context(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Create a prompt with context from retrieved documents"""
    if context_docs:
        # Format context with source information
        context_sections = []
        for i, doc in enumerate(context_docs):
            source = doc.get("source", "Unknown")
            context_sections.append(f"[Document {i+1}] From: {source}\n{doc['text']}")
            
        context_text = "\n\n" + "\n\n".join(context_sections)
        
        prompt = f"""
You are a helpful AI assistant that answers user questions accurately and clearly based on the provided documents.

IMPORTANT INSTRUCTIONS:
1. Base your answer primarily on the information in the CONTEXT below
2. Use the specific information from the documents to provide concrete details
3. If the context contains relevant information, cite the document number in your answer like [Document 1], [Document 2], etc.
4. If the context doesn't contain sufficient information, state that clearly and provide a general response based on your knowledge
5. Answer directly and concisely with relevant details from the documents

CONTEXT:{context_text}

USER QUESTION: {query}

ANSWER:
"""
    else:
        # No context provided, use general knowledge
        prompt = f"""
You are a helpful AI assistant that answers user questions accurately and clearly.

No document context is available for this question, so please use your general knowledge to answer.

If you truly don't know the answer, say "I don't have enough information to answer that question accurately, but I can help you if you upload relevant documents."

USER QUESTION: {query}

ANSWER:
"""
    return prompt

def get_chat_response(prompt: str, model: str = LLM_MODEL) -> str:
    """Get a response from the Groq language model"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides accurate information based only on the context provided."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error in Groq API: {response.text}")
            return "I'm sorry, I encountered an error while generating a response."
            
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error getting chat response: {e}")
        return "I'm sorry, I encountered an error while generating a response."

def query_with_sources(query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Query the LLM with context and return the answer with sources"""
    prompt = create_prompt_with_context(query, context_docs)
    answer = get_chat_response(prompt)
    
    # Format sources
    sources = []
    for doc in context_docs:
        source = {
            "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
            "source": doc.get("source", "Unknown")
        }
        if source["source"] not in [s["source"] for s in sources]:
            sources.append(source)
    
    return {
        "answer": answer,
        "sources": sources
    } 