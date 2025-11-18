import argparse
import chromadb
from langchain_ollama import OllamaLLM as Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    from chromadb.config import Settings
    client = chromadb.Client(settings=Settings(is_persistent=True, persist_directory=CHROMA_PATH))
    try:
        collection = client.get_collection("documents")
    except Exception:
        collection = client.create_collection("documents")

    # Compute query embedding and query collection directly.
    q_emb = embedding_function(query_text)
    results = collection.query(query_embeddings=[q_emb], n_results=8, include=["documents", "metadatas", "distances"])

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    context_text = "\n\n---\n\n".join(docs)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    model = Ollama(model="llama3:latest")
    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        # If the Ollama model/service is not available, fall back to showing
        # the retrieved context and sources so the user still sees results.
        print("LLM call failed:", type(e).__name__, e)
        print("Top retrieved documents:")
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            snippet = doc[:500].replace('\n', ' ')
            print(f"[{i}] {meta.get('id')} - {snippet}...")
        sources = [m.get("id") for m in metadatas]
        print("Sources:", sources)
        return "".join(docs)

    sources = [m.get("id") for m in metadatas]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()