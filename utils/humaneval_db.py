#utils/humaneval_db.py
from datasets import load_dataset
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

def load_humaneval_data():
    data = load_dataset("openai/openai_humaneval")
    df = data["test"].to_pandas()
    return pd.DataFrame(df[["task_id", "prompt", "canonical_solution"]])

def init_chroma(db_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    try:
        collection = client.get_collection("humaneval")
        print("✅ Loaded existing ChromaDB collection.")
    except Exception:
        collection = client.create_collection("humaneval", embedding_function=ef)
        print("🆕 Created new ChromaDB collection.")
    return collection

def store_embeddings(collection):
    data = load_humaneval_data()
    existing = collection.count()
    if existing == 0:
        for i, (task_id, prompt, solution) in enumerate(zip(data["task_id"], data["prompt"], data["canonical_solution"])):
            text_to_embed = f"{prompt}\nSolution: {solution}"
            collection.add(
                ids=[str(i)],
                documents=[text_to_embed],
                metadatas=[{"task_id": task_id, "prompt": prompt, "canonical_solution": solution}]
            )
        print(f"✅ Stored {len(data)} documents in ChromaDB.")
    else:
        print(f"ℹ️ ChromaDB already contains {existing} entries — skipping embedding.")

def retrieve_similar(collection, query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas
