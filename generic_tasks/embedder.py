import json
import requests

class OllamaEmbedder:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "open-orca-platypus2")
        self.ollama_url = "http://localhost:11434/api/embeddings"

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed a list of documents

        :param documents:
        :return:
        """
        results = []
        for document in documents:
            results.append(self.embed_query(document))

        return results

    def embed_query(self, document: str) -> list[float]:
        """
        Embed a single document/query

        :param document:
        :return:
        """
        request = {
            "model": self.model,
            "prompt": document,
            "options": {
                "temperature": 0.0,
            },
            "stream": False,
        }

        response = requests.post(self.ollama_url, json=request)

        return json.loads(response.text)["embedding"]

if __name__ == "__main__":
    embedder = OllamaEmbedder()
    response = embedder.embed_documents(["This is a test", "This is another test"])

    num_results = len(response)
    print(f"Got {num_results} embeddings")

    len1 = len(response[0])
    len2 = len(response[1])
    assert len1 == len2, f"Got different lengths: {len1} and {len2}"

    ## Assert that the embeddings are floats
    for embedding in response:
        assert isinstance(embedding[0], float), "Embedding is not a float"

    print("Each embedding has a length of " + str(len1) + " floats")
