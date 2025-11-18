from fastembed import TextEmbedding
import numpy as np

def cosine(a, b): 
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main():
    try:
        print("Loading model (may download files)...")
        model = TextEmbedding("BAAI/bge-base-en-v1.5")

        texts = [
            "Hello world",
            "Hi there",
            "This is a completely different sentence about quantum physics."
        ]
        print("Embedding", len(texts), "texts...")
        vecs = model.embed(texts)
        vecs = list(vecs)  # Coerce generator to list
        vecs = np.asarray(vecs, dtype=float)

        print("Embeddings shape:", vecs.shape)
        print("First embedding (first 10 dims):", vecs[0][:10].tolist())

        print("\nPairwise cosine similarities:")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = cosine(vecs[i], vecs[j])
                print(f"  [{i}] vs [{j}] = {sim:.4f}  --  \"{texts[i]}\"  |  \"{texts[j]}\"")

        print("\nSanity checks:")
        print("  dtype:", vecs.dtype)
        print("  any NaN:", np.isnan(vecs).any())
        print("  norms (first 3):", np.linalg.norm(vecs, axis=1)[:3].tolist())

    except Exception as e:
        print("ERROR running embedding test:", type(e).__name__, e)

if __name__ == "__main__":
    main()