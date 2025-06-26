import os
import hashlib
import pypdf
import pandas as pd
import docx
import pytesseract
import tiktoken
import faiss
import numpy as np
import pickle
from pdf2image import convert_from_path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()


# --------------------------
# Azure OpenAI Embedding Client
# --------------------------
class AzureOpenAIEmbeddings:
    def __init__(self):
        self.api_key = os.getenv('AZURE_OPENAI_API_EMBEDDING_KEY')
        self.api_version = os.getenv('AZURE_EMBEDDING_API_VERSION')
        self.endpoint = os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT')
        self.deployment_name = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment_name
        )
        return [item.embedding for item in response.data]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from a sample text."""
        sample_embedding = self.get_embeddings(["sample text"])[0]
        return len(sample_embedding)


# --------------------------
# FAISS Vector Database Manager
# --------------------------
class FAISSVectorDB:
    def __init__(self, index_path: str = "./vector_db/faiss_index.bin", chunks_path: str = "./vector_db/chunks.pkl"):
        self.index_path = index_path
        self.chunks_path = chunks_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Initialize or load existing index
        self.index = None
        self.chunks = []  # Store all document chunks with metadata

        self.load_index()

    def load_index(self):
        """Load existing FAISS index and chunks."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f"Loaded existing FAISS index: {self.index_path}")

            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"Loaded chunks: {len(self.chunks)} entries")

        except Exception as e:
            print(f"Error loading existing data: {e}")
            # Initialize empty structures if loading fails
            self.index = None
            self.chunks = []

    def save_index(self):
        """Save FAISS index and chunks."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)

            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)

            print(f"Saved FAISS index and chunks")
        except Exception as e:
            print(f"Error saving index: {e}")

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add embeddings with metadata and IDs to the FAISS index."""
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Initialize index if it doesn't exist
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add to FAISS index
        start_idx = len(self.chunks)
        self.index.add(embeddings_array)

        # Add chunks with metadata
        for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
            chunk_data = {
                'id': doc_id,
                'metadata': metadata,
                'index': start_idx + i
            }
            self.chunks.append(chunk_data)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        if self.index is None or len(self.chunks) == 0:
            return []

        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        scores, indices = self.index.search(query_array, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk_data = self.chunks[idx]
                result = {
                    'id': chunk_data['id'],
                    'score': float(score),
                    'metadata': chunk_data['metadata']
                }
                results.append(result)

        return results

    def get_by_filter(self, filter_key: str, filter_value: str) -> List[str]:
        """Get document IDs by metadata filter."""
        matching_ids = []
        for chunk in self.chunks:
            if chunk['metadata'].get(filter_key) == filter_value:
                matching_ids.append(chunk['id'])
        return matching_ids

    def document_exists(self, file_path: str, file_hash: str) -> bool:
        """Check if a document with the same hash already exists."""
        for chunk in self.chunks:
            metadata = chunk['metadata']
            if metadata.get('path') == file_path and metadata.get('file_hash') == file_hash:
                return True
        return False


# --------------------------
# Token-aware Chunking
# --------------------------
def split_text_into_token_chunks(text: str, max_tokens: int = 500, model: str = "text-embedding-3-small") -> List[str]:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


# --------------------------
# Generate Unique ID (using file content hash)
# --------------------------
def generate_file_hash(file_path: str) -> str:
    """Generate a hash of the file content to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_chunk_id(file_path: str, chunk_num: int) -> str:
    """Generate a unique ID for a chunk."""
    return hashlib.md5(f"{file_path}_{chunk_num}".encode()).hexdigest()


# --------------------------
# OCR-based PDF text extractor
# --------------------------
def extract_text_with_ocr(pdf_path: str) -> str:
    images = convert_from_path(pdf_path, dpi=300)
    text = " ".join([pytesseract.image_to_string(img) for img in images])
    return text


# --------------------------
# Main Embedding Function
# --------------------------
def create_embeddings(folder_paths: List[str], index_path: str = "./vector_db/faiss_index.bin",
                      chunks_path: str = "./vector_db/chunks.pkl"):
    embeddings_client = AzureOpenAIEmbeddings()
    vector_db = FAISSVectorDB(index_path, chunks_path)

    for folder_path in folder_paths:
        # Handle single file case
        if os.path.isfile(folder_path):
            all_files = [folder_path]
        else:
            all_files = [os.path.join(root, f)
                         for root, _, files in os.walk(folder_path)
                         for f in files]

        for file_path in tqdm(all_files, desc="Processing files"):
            file = os.path.basename(file_path)
            file_hash = generate_file_hash(file_path)

            # Skip if the file is already processed (unchanged)
            if vector_db.document_exists(file_path, file_hash):
                print(f"⏩ Skipping (already processed): {file_path}")
                continue

            text = ""
            try:
                # PDF
                if file.endswith(".pdf"):
                    try:
                        with open(file_path, "rb") as f:
                            reader = pypdf.PdfReader(f)
                            text = " ".join([page.extract_text() or "" for page in reader.pages])
                    except Exception:
                        text = extract_text_with_ocr(file_path)

                # Text / Markdown
                elif file.endswith((".txt", ".md")):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                # Excel
                elif file.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                    text = df.to_string()

                # Word
                elif file.endswith(".docx"):
                    try:
                        doc = docx.Document(file_path)
                        text = " ".join([para.text for para in doc.paragraphs])
                    except Exception:
                        text = pytesseract.image_to_string(file_path)

                if text.strip():
                    chunks = split_text_into_token_chunks(text, max_tokens=500)

                    # Process chunks in batches for efficiency
                    batch_size = 10
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i + batch_size]
                        batch_embeddings = embeddings_client.get_embeddings(batch_chunks)

                        batch_metadatas = []
                        batch_ids = []

                        for j, chunk in enumerate(batch_chunks):
                            chunk_num = i + j
                            doc_id = generate_chunk_id(file_path, chunk_num)
                            metadata = {
                                "source": file,
                                "path": file_path,
                                "chunk_num": chunk_num,
                                "file_hash": file_hash,
                                "text": chunk  # Store the actual text for retrieval
                            }
                            batch_metadatas.append(metadata)
                            batch_ids.append(doc_id)

                        vector_db.add_embeddings(batch_embeddings, batch_metadatas, batch_ids)

                    print(f"✅ Processed: {file_path} ({len(chunks)} chunks)")

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

    # Save the index after processing all files
    vector_db.save_index()
    print(f"💾 Saved FAISS index and chunks")


# --------------------------
# Search Function
# --------------------------
def search_documents(query: str, k: int = 5, index_path: str = "./vector_db/faiss_index.bin",
                     chunks_path: str = "./vector_db/chunks.pkl"):
    """Search for similar documents using a text query."""
    embeddings_client = AzureOpenAIEmbeddings()
    vector_db = FAISSVectorDB(index_path, chunks_path)

    # Get query embedding
    query_embedding = embeddings_client.get_embeddings([query])[0]

    # Search
    results = vector_db.search(query_embedding, k)

    return results


# --------------------------
# Run as script
# --------------------------
if __name__ == "__main__":
    # Example usage
    crop_folders = [
        r"Sugarcane_JAIN Irrigation.pdf",
    ]

    # Create embeddings
    create_embeddings(crop_folders)
