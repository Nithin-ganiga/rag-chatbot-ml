# Retrieval Augmented Generation (RAG) Project Notes

## 1. Data Preparation

### Loading a CSV file into Pandas
```python
import pandas as pd

# Load the CSV file
df = pd.read_csv('your_file.csv')

# Drop empty fields
df = df.dropna()
```
**Explanation:**
- `import pandas as pd`: Imports the Pandas library for data manipulation.
- `pd.read_csv('your_file.csv')`: Loads data from a CSV file into a DataFrame.
- `df.dropna()`: Removes any rows with empty (NaN) values to ensure data quality.

---

## 2. Vector Database Setup

### Importing Qdrant and SentenceTransformers
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
```
**Explanation:**
- `QdrantClient`: Used to interact with the Qdrant vector database.
- `SentenceTransformer`: Loads a pre-trained model (here, 'all-MiniLM-L6-v2') for encoding text into vector embeddings.
- `model = SentenceTransformer('all-MiniLM-L6-v2')`: Initializes the transformer model for embedding text data.

---

## 3. Creating a Collection in Qdrant

```python
# Initialize Qdrant client
client = QdrantClient()

# Create a collection
client.recreate_collection(
    collection_name='top_winds',
    vector_size=384,  # Size based on the model used
    distance='Cosine'
)
```
**Explanation:**
- `QdrantClient()`: Initializes a client instance to connect to Qdrant.
- `recreate_collection`: Creates (or resets) a collection named 'top_winds' in Qdrant.
- `vector_size=384`: The dimension of vectors produced by the embedding model.
- `distance='Cosine'`: Sets cosine similarity as the distance metric for vector comparison.

---

## 4. Encoding Data and Uploading to Vector Database

```python
# Encode the data
embeddings = model.encode(df['notes'].tolist())

# Upload data to the collection
client.upload_collection(
    collection_name='top_winds',
    vectors=embeddings,
    payload=df.to_dict(orient='records')
)
```
**Explanation:**
- `model.encode(df['notes'].tolist())`: Converts the 'notes' column of the DataFrame into vector embeddings.
- `client.upload_collection(...)`: Uploads the embeddings and original data (as payload) to the Qdrant collection for storage and retrieval.

---

## 5. Searching the Collection

```python
# Perform a search
results = client.search(
    collection_name='top_winds',
    query_vector=model.encode(['your search query']),
    limit=3
)
```
**Explanation:**
- `model.encode(['your search query'])`: Converts the user query into a vector for searching.
- `client.search(...)`: Searches the 'top_winds' collection for vectors most similar to the query, returning the top 3 matches.

---

# Retrieval Augmented Generation (RAG) Project Notes

## 4. Integrating Wine Data with a Large Language Model (LLM)

### 1. Importing Libraries
```python
import pandas as pd
```
**Explanation:**  
- Imports Pandas, a powerful library for data manipulation and analysis.

---

### 2. Loading Data
- The wine dataset is loaded and reduced to 700 entries for efficient processing.
- This helps speed up operations and model integration.

---

### 3. Creating a Vector Database Client
```python
from qdrant_client import QdrantClient
client = QdrantClient()
```
**Explanation:**  
- Imports QdrantClient and initializes a client instance for interacting with the vector database.

---

### 4. Storing Wine Data
```python
# Create a collection to store wines
client.create_collection("top_wines")
```
**Explanation:**  
- Creates a new collection in Qdrant, named "top_wines", to store the wine data.

---

### 5. Uploading Data
```python
# Uploading the wine data into the collection
client.upload_collection("top_wines", wine_data)
```
**Explanation:**  
- Uploads the prepared wine data into the "top_wines" collection in the vector database.

---

### 6. Connecting to OpenAI API
```python
import openai
openai.api_key = 'your-api-key'
```
**Explanation:**  
- Imports the OpenAI library and sets the API key to enable requests to the language model.

---

### 7. Creating a Prompt
```python
prompt = "Suggest me an amazing Malbec wine from Argentina."
```
**Explanation:**  
- Sets up a user prompt to be sent to the LLM for recommendations.

---

### 8. Getting a Response from the Model
```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
```
**Explanation:**  
- Sends the prompt to the OpenAI chat model and retrieves the generated response.

---

### 9. Displaying the Result
- The model suggests a wine from the database based on the user's prompt.

---

**Summary:**  
This workflow demonstrates how to efficiently import, store, and query wine data with Pandas and Qdrant, and how to integrate this setup with OpenAI's GPT model to provide intelligent wine recommendations.

