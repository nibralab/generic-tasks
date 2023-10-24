import torch
from transformers import pipeline


def sentiment_analysis(utterance: str):
    """
    Generic Sentiment Analysis

    :param utterance: The text to be analyzed
    :return: The sentiment (str) and the confidence level (float)
    """

    model = "distilbert-base-uncased-finetuned-sst-2-english"

    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    prediction = classifier(utterance)

    sentiment = prediction[0]['label']
    score = prediction[0]['score']

    return sentiment, score