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

    llm = Ollama(
        base_url=ollama_url,
        model=model,
    )

    # Get the answer from the chain
    answer = llm(
        text,
        max_length=max_length,
        num_return_sequences=num_return_sequences
    )

    return answer
