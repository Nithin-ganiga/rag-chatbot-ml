# SynthiQuery - AI PDF Chat

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents using AI.

![Streamlit App Screenshot](https://via.placeholder.com/800x450.png?text=SynthiQuery+Screenshot)

## Features

- 📄 **Document Upload**: Upload and process PDF documents
- 🔍 **Semantic Search**: Find relevant information across all your documents
- 💬 **Chat Interface**: Ask questions about your documents in natural language
- 🔗 **Source Citations**: See which documents and passages answers come from
- 📊 **Beautiful UI**: Clean, intuitive Streamlit interface
- 🆓 **Free APIs**: Uses Groq and Hugging Face free APIs instead of OpenAI
- 📂 **File Index**: Browse and manage your uploaded documents
- 🧹 **Auto Cleanup**: Automatically cleans up data when session ends

## Quick Start

### Option 1: Run with Python

1. **Clone the repository**
   ```bash
   git clone https://github.com/nithin-ganiga/rag-chatbot-ml.git
   cd rag-chatbot-ml
   ```

2. **Set up environment**
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   # Copy the example env file
   cp env.example .env
   
   # Edit the .env file with your API keys
   # On Windows:
   notepad .env
   
   # On macOS/Linux:
   nano .env
   ```

4. **Run the application**
   ```bash
   python run.py
   # Or directly with Streamlit
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will be available at http://localhost:8501

### Option 2: Run with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/nithin-ganiga/rag-chatbot-ml.git
   cd rag-chatbot-ml
   ```

2. **Configure environment variables**
   ```bash
   # Copy the example env file
   cp env.example .env
   
   # Edit the .env file with your API keys
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Open your browser**
   - The app will be available at http://localhost:8501

## How to Use

1. **Upload Documents**:
   - Use the sidebar to upload PDF documents
   - The system will process the document and add it to the vector database

2. **Chat with Your Documents**:
   - Type your questions in the chat input at the bottom
   - The AI will respond with answers based on your documents
   - Sources will be displayed below each answer

3. **Manage Files**:
   - Use the "File Index" tab to view all uploaded files
   - Delete individual files or all files at once
   - See file details like size and upload date

4. **Reset Database**:
   - Use the "Reset Document Database" button in the sidebar to clear all documents
   - This will also clean up all uploaded files

5. **End Session**:
   - Use the "End Session and Clean Up" button to manually trigger data cleanup
   - Data is automatically cleaned when the app exits

## API Keys

This project uses two free API services:

1. **Groq API**: For text generation (LLM)
   - Sign up at [console.groq.com](https://console.groq.com)
   - Create an API key and add it to your `.env` file as `GROQ_API_KEY`

2. **Hugging Face API**: For embeddings 
   - Sign up at [huggingface.co](https://huggingface.co)
   - Create an API key and add it to your `.env` file as `HF_API_KEY`

## Project Structure

```
rag-chatbot-ml/
├── app.py                # Streamlit application
├── run.py                # Helper script to run the application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── env.example           # Example environment variables
├── data/                 # Data storage
│   ├── raw/              # Raw PDF files
│   └── processed/        # Processed data and vector store
└── src/                  # Source code
    ├── config.py         # Configuration settings
    ├── ingestion/        # PDF loading and processing
    ├── embedding/        # Text embedding functionality
    ├── retrieval/        # Vector store and retrieval
    ├── llm/              # LLM integration
    └── utils/            # Utility functions
```

## How It Works

1. **Document Processing**: PDF files are uploaded, text is extracted and split into chunks.
2. **Embedding**: Text chunks are converted to vector embeddings using Hugging Face's embedding models.
3. **Storage**: Embeddings are stored in a ChromaDB vector database.
4. **Retrieval**: When a question is asked, the system finds the most relevant chunks.
5. **Generation**: An LLM (Groq's LLaMA model) uses the retrieved chunks as context to generate an answer.
6. **Data Management**: Files are indexed and can be managed through the UI.
7. **Auto-Cleanup**: The system automatically cleans up when a session ends.

## Deployment

### Deploy to a server

1. SSH into your server
2. Clone the repository
3. Create a `.env` file with your API keys
4. Run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add your API keys as secrets
5. Deploy!

## Troubleshooting

- **API Key Issues**: Ensure your API keys are correctly set in your .env file
- **PDF Extraction Failures**: The system uses multiple PDF extraction methods as fallbacks
- **Vector Store Errors**: Try resetting the database using the reset button
- **Data Not Being Cleaned**: Try the "End Session and Clean Up" button in the sidebar

### Vector Store Issues

If you're experiencing issues with the vector store not properly indexing documents or retrieving results:

1. **Environment Variables**: Make sure your `.env` file is properly encoded in UTF-8 and contains all required variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   HF_API_KEY=your_huggingface_api_key
   RAW_DATA_DIR=./data/raw
   COLLECTION_NAME=pdf_documents
   ```

2. **HuggingFace API Permissions**: The app includes a local fallback for embedding generation if the HuggingFace API fails.

3. **Path Issues**: The app now handles path issues on Windows systems.

4. **Diagnostics**: Use the Diagnostics tab to check vector DB health, run test queries, and diagnose issues.

5. **Reset Database**: If issues persist, try using the "Reset Document Database" button in the sidebar.

### File Upload Issues

If document uploads aren't working correctly:

1. Check the console logs for any errors during upload processing
2. Ensure the RAW_DATA_DIR path exists and is writable
3. Try a smaller PDF file (less than 5MB) if you're having trouble with large files

## License

MIT

## Acknowledgements

- Groq for providing the free LLM API
- Hugging Face for the free embedding API
- LangChain for the document processing tools
- Streamlit for the web interface
