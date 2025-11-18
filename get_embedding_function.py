from fastembed import TextEmbedding


def get_embedding_function(model_name: str = "BAAI/bge-base-en-v1.5"):
    """Return a callable embedding function compatible with LangChain/Chroma.

    The returned callable accepts a single string or a list of strings and
    returns a list of embedding vectors (or a single vector for a single
    input string).
    """
    model = TextEmbedding(model_name)

    def embed(texts):
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        vecs = list(model.embed(texts))

        if single:
            return vecs[0]
        return vecs

    return embed
