import requests
from langchain.llms import Ollama

def complete(text: str, max_length: int = 50, num_return_sequences: int = 1):
    """
    Generic Text Completion

    :param text: The text to be completed
    :param max_length:
    :param num_return_sequences:
    :return: The completed text
    """

    print(f"Complete: {text}")
    ollama_url = "http://localhost:11434/api/generate"
    model = "llama2-uncensored"

    request = {
        "model": model,
        "prompt": text,
        "stream": False,
    }

    # Send the request to the Ollama URL
    response = requests.post(ollama_url, json=request)

    # Get the response as JSON
    response = response.json()

    # Get the answer from the response
    answer = response["response"]

    return answer
