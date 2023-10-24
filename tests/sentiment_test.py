import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from generic_tasks import sentiment_analysis

texts = [
    "I am very happy with the product.",
    "I am not happy with the product.",
    "The product works as expected.",
]

for text in texts:
    sentiment, confidence = sentiment_analysis(text)
    percentage = round(confidence * 100, 2)
    print(f"{text}")
    print(f"-> {sentiment} (Confidence: {percentage}%)\n")
