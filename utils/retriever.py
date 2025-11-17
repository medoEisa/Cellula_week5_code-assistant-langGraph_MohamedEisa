# utils/retriever.py
from utils.humaneval_db import retrieve_similar

class Retriever:
    def __init__(self, collection):
        self.collection = collection

    def retrieve(self, query: str, top_k: int = 3):
        docs, metas = retrieve_similar(self.collection, query, top_k=top_k)
        examples = []
        if not docs or not metas:
            return examples
        for doc, meta in zip(docs, metas):
            if meta is None:
                meta = {}
            examples.append({
                "prompt": meta.get("prompt", "") if isinstance(meta, dict) else "",
                "canonical_solution": meta.get("canonical_solution", "") if isinstance(meta, dict) else "",
                "task_id": meta.get("task_id", "") if isinstance(meta, dict) else ""
            })
        return examples
