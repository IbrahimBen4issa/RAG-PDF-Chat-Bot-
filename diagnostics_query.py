import chromadb
from get_embedding_function import get_embedding_function

Q = "give list of different peripheralson the board like TIM and ADC"

def main():
    client = chromadb.Client()
    try:
        collection = client.get_collection("documents")
    except Exception as e:
        print("No collection 'documents' found:", e)
        return

    # Try to get a small sample of stored items
    try:
        sample = collection.get(include=["ids", "metadatas", "documents"], limit=5)
        print("Sample keys:", list(sample.keys()))
        print("Sample ids:", sample.get("ids"))
        print("Sample metadatas:", sample.get("metadatas"))
    except Exception as e:
        print("Error reading collection sample:", e)

    # Print total count if available
    cnt = None
    if hasattr(collection, "count"):
        try:
            cnt = collection.count()
        except Exception:
            cnt = None
    print("Collection count:", cnt)

    # Compute query embedding
    embed_fn = get_embedding_function()
    try:
        q_emb = embed_fn(Q)
    except Exception as e:
        print("Embedding function error:", e)
        return

    print("Query embedding length:", len(q_emb) if q_emb is not None else None)

    # Run a raw query
    try:
        results = collection.query(query_embeddings=[q_emb], n_results=5, include=["documents", "metadatas", "ids", "distances"])
        print("Query raw result keys:", list(results.keys()))
        print(results)
    except Exception as e:
        print("Query failed:", e)

if __name__ == '__main__':
    main()
