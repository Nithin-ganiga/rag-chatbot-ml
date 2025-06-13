import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import uuid
from typing import List, Dict, Any, BinaryIO
from pathlib import Path

# Add the parent directory to sys.path to ensure proper imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import after adding to sys.path
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Define RAW_DATA_DIR with a fallback
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
print(f"Using RAW_DATA_DIR: {RAW_DATA_DIR}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2 and pdfplumber as fallback"""
    try:
        # Try using PyPDF2 first
        import PyPDF2
        text = ""
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ""
        
        if text.strip():
            return text
        
        # If PyPDF2 returns empty text, try pdfplumber
        return extract_text_with_pdfplumber(pdf_path)
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")
        return extract_text_with_pdfplumber(pdf_path)

def extract_text_with_pdfplumber(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text with pdfplumber: {e}")
        return ""

def chunk_text(text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    
    # Add metadata to each chunk
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "text": chunk,
            "chunk_id": i,
        }
        
        # Add metadata if provided
        if metadata:
            doc.update(metadata)
            
        documents.append(doc)
    
    return documents

def process_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
    """Process a PDF file: extract text and split into chunks with metadata"""
    text = extract_text_from_pdf(file_path)
    
    if not text:
        print(f"Could not extract text from {filename}")
        return []
    
    # Create metadata for the document
    metadata = {
        "source": filename,
        "file_path": file_path,
    }
    
    # Chunk the text
    chunks = chunk_text(text, metadata)
    
    print(f"Processed {filename}: {len(chunks)} chunks created")
    return chunks

def save_uploaded_pdf(uploaded_file: BinaryIO) -> str:
    """Save an uploaded PDF file to the raw data directory"""
    # Generate a unique filename if one is not provided
    if hasattr(uploaded_file, 'name'):
        filename = uploaded_file.name
    else:
        filename = f"uploaded_{uuid.uuid4()}.pdf"
    
    # Ensure file path is created properly
    file_path = os.path.join(RAW_DATA_DIR, filename)
    
    # Write the file
    with open(file_path, "wb") as f:
        if hasattr(uploaded_file, 'read'):
            f.write(uploaded_file.read())
        elif hasattr(uploaded_file, 'getbuffer'):
            f.write(uploaded_file.getbuffer())
        else:
            f.write(uploaded_file)
    
    return file_path 