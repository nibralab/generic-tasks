from typing import List

import torch
from transformers import pipeline, AutoTokenizer, BartForSequenceClassification


def classify(text: str, labels: List):
    """
    Generic Zero-Shot Classification

    :param text: The text to be classified
    :param labels: The classes to be evaluated

    :return The label for most likely classification (str) and the confidence level (float)
    """
    model_name = "ModelTC/bart-base-mnli"
    model = BartForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    prediction = classifier(
        text,
        labels,
        truncation=False,
        padding=True,
        return_tensors="pt"
    )

    candidates = [{'label': label, 'score': score} for label, score in zip(prediction['labels'], prediction['scores'])]

    strongest_candidate = max(candidates, key=lambda x: x['score'])
    classification = strongest_candidate['label']
    confidence = strongest_candidate['score']

    return classification, confidence
