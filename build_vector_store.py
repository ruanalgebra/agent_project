"""
Knowledge base builder for RAG applications.

Loads text documents from a directory, splits them into chunks,
generates embeddings using Ollama, and stores them in a Chroma vector database.
"""

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ---------- Custom Loader for UTF-8 Encoding ----------
class UTF8TextLoader(TextLoader):
    """TextLoader with explicit UTF-8 encoding support."""
    def __init__(self, file_path: str):
        super().__init__(file_path, encoding="utf-8")


# ---------- Configuration ----------
DATA_DIR = "./knowledge_base"          # Directory containing source documents
FILE_PATTERN = "**/*.txt"              # File pattern to match
CHUNK_SIZE = 500                       # Tokens per chunk
CHUNK_OVERLAP = 50                     # Overlap between chunks
EMBEDDING_MODEL = "nomic-embed-text"   # Ollama embedding model
PERSIST_DIR = "./chroma_db"            # Vector database persistence directory


# ---------- Build Knowledge Base ----------
def build_knowledge_base(
    data_dir: str = DATA_DIR,
    pattern: str = FILE_PATTERN,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model: str = EMBEDDING_MODEL,
    persist_dir: str = PERSIST_DIR,
) -> int:
    """
    Load documents, split into chunks, and store as vector embeddings.

    Returns:
        int: Number of chunks created.
    """
    # 1. Load documents
    loader = DirectoryLoader(
        data_dir,
        glob=pattern,
        loader_cls=UTF8TextLoader,
    )
    docs = loader.load()
    if not docs:
        print(f"⚠️ No documents found in {data_dir}")
        return 0

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # 3. Generate embeddings and store in Chroma
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir,
    )

    return len(chunks)


# ---------- Main ----------
if __name__ == "__main__":
    print("🚀 Building knowledge base...")
    print(f"📂 Source: {DATA_DIR} -> Pattern: {FILE_PATTERN}")
    print(f"⚙️  Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"🧠 Embedding model: {EMBEDDING_MODEL}")

    count = build_knowledge_base()

    if count > 0:
        print(f"✅ Knowledge base built successfully with {count} chunks.")
        print(f"💾 Vector database saved to {PERSIST_DIR}")
    else:
        print("❌ Build failed. Please check your data directory.")