
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append("src")

from dnd_rag.core.retriever import Retriever # type: ignore
from dnd_rag.providers.embeddings import embed_texts # type: ignore
from dnd_rag.core.config import load_ingest_config, DEFAULT_CONFIG_PATH # type: ignore

async def main():
    question = "Как удержать интерес игрока-актёра?"
    expected_id = "dmg_ch06_0011"
    
    print(f"Question: {question}")
    print(f"Expected Chunk ID: {expected_id}")
    
    # Load config
    cfg = load_ingest_config(DEFAULT_CONFIG_PATH)
    
    # Initialize Retriever
    retriever = Retriever(collection="dnd_rule_assistant")
    
    # Embed question
    print("Embedding question...")
    query_vec = embed_texts([question], model="text-embedding-3-small")[0]
    
    # Search
    print("Searching...")
    results = await retriever.search(query_vec, limit=10)
    
    print(f"Found {len(results)} chunks:")
    found = False
    for i, chunk in enumerate(results):
        print(f"[{i+1}] ID: {chunk.chunk_id}, Score: {chunk.score:.4f}")
        print(f"    Text: {chunk.text[:100]}...")
        if chunk.chunk_id == expected_id:
            found = True
            print("    *** MATCHED EXPECTED ID ***")
            
    if not found:
        print(f"\nExpected ID {expected_id} NOT found in top 10.")
        
if __name__ == "__main__":
    asyncio.run(main())
