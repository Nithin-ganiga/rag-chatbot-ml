import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Force UTF-8 encoding for .env file
env_path = BASE_DIR / '.env'
print(f"Looking for .env at: {env_path}")

# First try to load the .env file
env_loaded = False
if env_path.exists():
    print(f".env file found at {env_path}")
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            env_content = f.read()
            print(f"Read {len(env_content)} characters from .env file")
        
        # Load from the existing .env file
        load_dotenv(dotenv_path=env_path)
        env_loaded = True
        print("Environment variables loaded from .env file")
    except Exception as e:
        print(f"Error reading .env file: {e}")
        # Try the fallback
        try:
            load_dotenv()
            env_loaded = True
            print("Environment variables loaded using fallback method")
        except Exception as e2:
            print(f"Error loading environment variables: {e2}")

if not env_loaded:
    print("Could not load environment variables from .env file")
    # Try fallback
    load_dotenv()

# Get RAW_DATA_DIR from .env or use default if not set
raw_data_dir_env = os.getenv("RAW_DATA_DIR")
if raw_data_dir_env is None or not raw_data_dir_env:
    print("RAW_DATA_DIR not set in environment. Using default path.")
    RAW_DATA_DIR = DATA_DIR / "raw"
else:
    # Convert to Path object 
    # If it's a relative path starting with ./, use BASE_DIR as the base
    if raw_data_dir_env.startswith('./'):
        RAW_DATA_DIR = BASE_DIR / raw_data_dir_env[2:]
    else:
        RAW_DATA_DIR = Path(raw_data_dir_env)

# Print paths for debugging
print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
print(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Vector DB settings
VECTOR_DB_PATH = PROCESSED_DATA_DIR / "vector_store"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "pdf_documents")
print(f"VECTOR_DB_PATH: {VECTOR_DB_PATH}")
print(f"COLLECTION_NAME: {COLLECTION_NAME}")

# Ensure vector store directory exists
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# PDF Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# API Keys with fallbacks
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

# Check if keys are set
print(f"GROQ_API_KEY set: {'Yes' if GROQ_API_KEY else 'No'}")
print(f"HF_API_KEY set: {'Yes' if HF_API_KEY else 'No'}")

# Model settings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3-70b-8192")

# Application settings
APP_TITLE = "SynthiQuery"
APP_DESCRIPTION = "Answer questions about the PDF documents"
