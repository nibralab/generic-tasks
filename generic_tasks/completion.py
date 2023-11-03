import requests

def complete(text: str, **kwargs):
    """
    Generic Text Completion

    :param text: The text to be completed
    :return: The completed text
    """

    model = kwargs.get("model", "open-orca-platypus2")

    ollama_url = "http://localhost:11434/api/generate"
    request = {
        "model": model,
        "prompt": f"Complete the following test:\n\nText: {text}",
        "stream": False,
    }

    # Send the request to the Ollama URL
    response = requests.post(ollama_url, json=request)

    # Get the response as JSON
    response = response.json()

    # Get the answer from the response
    answer = response["response"]

    return answer

if __name__ == "__main__":
    models = [
        "mistral",
        "llama2",
        "codellama",
        "vicuna",
        "orca-mini",
        "llama2-uncensored",
        "wizard-vicuna-uncensored",
        "nous-hermes", # education 56.2%, politics 100.0%, education 56.2% ✘
        "phind-codellama",
        "mistral-openorca",
        "wizardcoder",
        "wizard-math",
        "llama2-chinese",
        "stable-beluga",
        "codeup",
        "everythinglm",
        "medllama2",
        "wizardlm-uncensored",
        "zephyr",
        "falcon",
        "wizard-vicuna",
        "open-orca-platypus2", # education 95.2%, politics 100.0%, business 85.7% ✔
        "starcoder",
        "samantha-mistral",
        "wizardlm",
        "sqlcoder",
        "dolphin2.1-mistral",
        "nexusraven",
        "openhermes2-mistral",
        "dolphin2.2-mistral",
        "codebooga",
    ]

    text = "In this course, we will teach you how to"

    for model in models:
        completed_text = complete(text, model=model)
        print(f"{model}: {text} {completed_text}")
