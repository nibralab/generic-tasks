import json
import re
from typing import List

import requests

def fix_json(text):
    """
    Fix JSON from Ollama

    :param text: The text to be fixed
    :return: The fixed text
    """
    json_pattern = r"(?:```(?:json)?)(.*?)```"

    if not re.search(json_pattern, text, re.DOTALL|re.MULTILINE):
        ollama_url = "http://localhost:11434/api/generate"
        request = {
            "model": "open-orca-platypus2",
            "prompt": f"This is a malformed JSON string:\n\n{text}\n\nFix it to have correct JSON syntax. "
                      "Add brackets, braces, commas as needed to make it well-formed. "
                      "Prefer List over Dict, if possible.\n\n```json\n",
            "options": {
                "temperature": 0.0,
            },
            "stream": False,
        }

        response = requests.post(ollama_url, json=request)

        try:
            prediction = json.loads(response.text)
        except json.decoder.JSONDecodeError:
            print(f"JSONDecodeError: {response.text}")
            return "Parse error", 0.0

        text = prediction["response"]

    fixed_json = re.findall(json_pattern, text, re.DOTALL)[0]

    return fixed_json


def classify(text: str, labels: List, **kwargs):
    """
    Generic Zero-Shot Classification

    :param text: The text to be classified
    :param labels: The classes to be evaluated

    :return The label for most likely classification (str) and the confidence level (float)
    """

    model = kwargs.get("model", "open-orca-platypus2")

    ollama_url = "http://localhost:11434/api/generate"
    request = {
        "model": model,
        "system": "You are an excellent classifier. "
                  "You are given a text and a list of labels. "
                  "You must state, how confident you are, that the text belongs to the label. "
                  "Return your answer in pure JSON format as a dict, with the keys 'label' and 'score'.\n\n"
                  "Example: {\"label\": \"politics\", \"score\": 0.789}\n\n"
                  "Don't add any explanation, just return the bare results. ",
        "prompt": f"Text: »{text}«\nLabels: {', '.join(labels)}",
        "options": {
            "temperature": 0.0,
        },
        "stream": False,
    }
    response = None

    try:
        # Send the request to the Ollama URL
        response = requests.post(ollama_url, json=request)

        if response == "<Response [404]>":
            return f"Model {model} is not available. Did you download it?", 0.0

        prediction = json.loads(response.text)

        try:
            prediction = json.loads(prediction["response"])
        except json.decoder.JSONDecodeError:
            print(f"JSONDecodeError: »{prediction['response']}«")
            try:
                fixed_json = fix_json(prediction["response"])
                prediction = json.loads(fixed_json)

            except json.decoder.JSONDecodeError:
                return "Parse error", 0.0

        # If prediction is a list, find the element with the largest score
        if isinstance(prediction, list):
            prediction = max(prediction, key=lambda x: x["score"])

        classification = prediction["label"]
        confidence = float(prediction["score"])

        return classification, confidence
    except Exception as e:
        print(f"Error {e}; response: {response}")
        return "Error", 0.0

if __name__ == "__main__":
    models = [
        # "mistral", # education 99.9%, politics 95.2%, education 12.3% ✘
        # "llama2", # education 85.7%, politics 85.0%, business 85.7% ✔
        # "codellama", # education 98.7%, politics 98.7%, business 78.9% ✔
        # "vicuna", # education 95.0%, politics 85.0%, business 95.0% ✔
        # "orca-mini", # education 78.9%, politics 78.9%, business 78.9% ✔ Occasionally JSON errors
        # "llama2-uncensored", # education 95.0%, politics 78.9%, business 95.0% ✔
        # "wizard-vicuna-uncensored", # education 87.5%, politics 78.9%, education/business 66.7% ✔ JSON errors
        "nous-hermes", # education 65.4%, politics 100.0%, education 56.2% ✔
        # "phind-codellama", # completely unusable
        # "mistral-openorca", # education 98.0%, politics 95.0%, politics 90.0% ✘
        # "wizardcoder", # completely unusable
        # "wizard-math",
        # "llama2-chinese",
        # "stable-beluga", # education 95.0%, politics 78.9%, business 78.9% ✔
        # "codeup",
        # "everythinglm", # education 78.9%, politics 78.9%, education 78.9% ✘ JSON errors
        # "medllama2",
        # "wizardlm-uncensored", # education 85.6%, politics 85.4%, education 0.0% ✘ JSON errors
        # "zephyr", # education 100.0%, politics 100.0%, none 12.3% ✘ JSON errors
        # "falcon", # completely unusable
        # "wizard-vicuna", # education 85.2%, politics 90.9%, business 56.3% ✔
        # "open-orca-platypus2", # education 95.2%, politics 100.0%, business 85.7% ✔
        # "starcoder",
        # "samantha-mistral",
        # "wizardlm", # Model not found
        # "sqlcoder",
        # "dolphin2.1-mistral", # education 95.0%, politics 95.0%, business 85.0% ✔
        # "nexusraven",
        # "openhermes2-mistral", # education 98.7%, politics 98.7%, business 98.7% ✔
        # "dolphin2.2-mistral", # education 100.0%, politics 95.0%, business 85.0% ✔
        # "codebooga",
    ]

    texts = [
        "This is a course about the Transformers library.",
        "The elections in the United States are over.",
        "The product works as expected.",
    ]
    labels = ["education", "politics", "business"]

    for model in models:
        print(f"\nModel: {model}")
        for text in texts:
            try:
                print(f"{text}")
                classification, confidence = classify(text, labels, model=model)
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except:
                print(f"Error in {model}")
                continue
            percentage = round(confidence * 100, 2)
            print(f"-> {classification} ({percentage}%)\n")
