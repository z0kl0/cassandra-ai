import os
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class FraudCorpusRAG:
    """
    Handles the local Vector Database (ChromaDB) for historical fraud cases.
    """
    def __init__(self):
        # Load path from .env, fallback to default if missing
        self.chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB PersistentClient at {self.chroma_path}")
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        
        # Using SentenceTransformers as specified in the proposal.
        # 'all-MiniLM-L6-v2' is a fast, lightweight model perfect for CPU inference.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name="historical_frauds",
            embedding_function=self.embedding_fn
        )

    def add_case(self, case_id: str, text: str, metadata: dict = None):
        """Adds a historical fraud case document to the vector database."""
        if metadata is None:
            metadata = {}
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[case_id]
        )
        logger.info(f"Added case {case_id} to ChromaDB.")

    def query_cases(self, query_text: str, n_results: int = 2) -> dict:
        """Queries the vector database for similar fraud cases."""
        logger.info(f"Querying ChromaDB for: '{query_text}'")
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    
    # Smoke test the RAG implementation
    rag = FraudCorpusRAG()
    rag.add_case(
        case_id="enron_001",
        text="Enron used mark-to-market accounting and special purpose entities (SPEs) to hide toxic assets and massive debts, inflating reported earnings.",
        metadata={"company": "Enron", "year": "2001", "fraud_type": "Off-balance sheet entities"}
    )
    
    print("\n--- Testing RAG Query ---")
    results = rag.query_cases("hidden debt and special purpose entities")
    
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}: {doc}")
        print(f"Metadata: {results['metadatas'][0][i]}")