import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import local modules
from src.retrieval.vector_store import VectorStore
from src.utils.helpers import get_answer, process_uploaded_file

def create_test_pdf():
    """Create a simple test PDF file in the raw data directory"""
    from reportlab.pdfgen import canvas
    from src.config import RAW_DATA_DIR
    
    pdf_path = os.path.join(RAW_DATA_DIR, "test_document.pdf")
    
    # Create a canvas
    c = canvas.Canvas(pdf_path)
    
    # Add some text
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "This is a test document about artificial intelligence.")
    c.drawString(100, 730, "AI systems are designed to perform tasks that typically require human intelligence.")
    c.drawString(100, 710, "These tasks include visual perception, speech recognition, and decision-making.")
    c.drawString(100, 690, "Machine learning is a subset of AI that enables systems to learn from data.")
    c.drawString(100, 670, "Deep learning is a type of machine learning that uses neural networks.")
    c.drawString(100, 650, "Natural Language Processing allows machines to understand human language.")
    c.drawString(100, 630, "Computer vision is the field of AI that enables machines to interpret visual information.")
    
    # Save the PDF
    c.save()
    
    print(f"Created test PDF at {pdf_path}")
    return pdf_path

def test_vector_store():
    print("Testing vector store functionality...")
    
    # Reset vector store
    vector_store = VectorStore()
    vector_store.reset()
    print("Vector store reset.")
    
    # Create a test PDF
    pdf_path = create_test_pdf()
    
    # Add document to vector store directly
    from src.ingestion.pdf_loader import process_pdf
    filename = os.path.basename(pdf_path)
    chunks = process_pdf(pdf_path, filename)
    
    if not chunks:
        print("Error: No chunks created from PDF")
        return False
        
    print(f"Created {len(chunks)} chunks from PDF.")
    print(f"Sample text: {chunks[0]['text'][:100]}...")
    
    # Add to vector store
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    # Verify document count
    count = vector_store.get_document_count()
    print(f"Document count after adding: {count}")
    
    # Search for a document
    results = vector_store.similar_search("What is artificial intelligence?", top_k=2)
    
    if results:
        print("Search successful!")
        print(f"Found {len(results)} results")
        print(f"First result: {results[0]['text'][:100]}...")
        
        # Test getting an answer
        answer_result = get_answer("What is artificial intelligence?")
        print("\nAI Answer:")
        print(answer_result.get("answer", "No answer found"))
        print("\nSources:")
        for source in answer_result.get("sources", []):
            if isinstance(source, dict):
                print(f"- {source.get('source', 'Unknown')}")
            else:
                print(f"- {source}")
        
        return True
    else:
        print("Search failed - no results found")
        return False

if __name__ == "__main__":
    success = test_vector_store()
    if success:
        print("\nVector store test PASSED ✓")
    else:
        print("\nVector store test FAILED ✗") 