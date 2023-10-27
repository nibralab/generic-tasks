import torch
from transformers import pipeline


def sentiment_analysis(utterance: str) -> (str, float):
    """
    Generic Sentiment Analysis

    :param utterance: The text to be analyzed
    :return: The sentiment (str) and the confidence level (float)
    """

    model = "SamLowe/roberta-base-go_emotions"

    classifier = pipeline(
        "text-classification",
        model=model,
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        top_k = None
    )

    # Produces a list of dicts for each of the labels
    emotions = classifier([utterance])[0]

    # Find the most likely label from the emotions dict
    prediction = max(emotions, key=lambda x: x['score'])

    return prediction['label'], prediction['score']
