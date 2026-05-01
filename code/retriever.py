import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from config import DATA_DIR, CHROMA_DB_DIR, OPENAI_API_KEY, EMBEDDING_MODEL

class Retriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        
        # We use OpenAI embeddings
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL
        ) if OPENAI_API_KEY else None
        
        # Create or get collections for each company
        self.collections = {
            "hackerrank": self.chroma_client.get_or_create_collection("hackerrank", embedding_function=self.embedding_fn),
            "claude": self.chroma_client.get_or_create_collection("claude", embedding_function=self.embedding_fn),
            "visa": self.chroma_client.get_or_create_collection("visa", embedding_function=self.embedding_fn),
        }

    def ingest_data(self):
        """Reads all markdown files from data/ and ingests them into ChromaDB."""
        if not self.embedding_fn:
            raise ValueError("OPENAI_API_KEY is not set. Cannot initialize embeddings.")
            
        for company in self.collections.keys():
            company_dir = DATA_DIR / company
            if not company_dir.exists():
                print(f"Directory not found: {company_dir}")
                continue
            
            collection = self.collections[company]
            
            # Check if already ingested to save time/money
            if collection.count() > 0:
                print(f"Collection '{company}' already has {collection.count()} documents. Skipping ingestion.")
                continue
                
            print(f"Ingesting documents for {company}...")
            documents = []
            metadatas = []
            ids = []
            
            # Recursively find all markdown files
            for md_file in company_dir.rglob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    # Basic chunking: split by 1500 characters
                    chunk_size = 1500
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                    
                    for idx, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({"source": str(md_file.relative_to(DATA_DIR)), "company": company})
                        ids.append(f"{md_file.stem}_{idx}")
                except Exception as e:
                    print(f"Error reading {md_file}: {e}")
            
            if not documents:
                print(f"No documents found for {company}.")
                continue
                
            # Add to collection in batches to avoid API limits
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                collection.add(
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
            print(f"Finished ingesting {len(documents)} chunks for {company}.")

    def retrieve(self, query: str, company: str = "None", top_k: int = 3) -> str:
        """Retrieves top_k relevant chunks. Searches specific company or all if None."""
        if not self.embedding_fn:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        results = []
        
        # Determine which collections to search
        company_lower = str(company).lower()
        companies_to_search = [company_lower] if company_lower in self.collections else self.collections.keys()
        
        for comp in companies_to_search:
            collection = self.collections[comp]
            if collection.count() == 0:
                continue
                
            res = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count())
            )
            
            if res["documents"] and res["documents"][0]:
                for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                    results.append(f"--- Document from {meta['source']} ---\n{doc}\n")
                
        # If searching across multiple, we return more chunks, let's limit the total text slightly
        # Or just return them all and let the LLM handle it
        return "\n".join(results[:top_k * len(companies_to_search)])

if __name__ == "__main__":
    # Test ingestion
    if OPENAI_API_KEY:
        retriever = Retriever()
        retriever.ingest_data()
    else:
        print("Set OPENAI_API_KEY in .env to test ingestion.")
