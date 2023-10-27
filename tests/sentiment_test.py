import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from generic_tasks import sentiment_analysis

import importlib
importlib.reload(sys.modules['generic_tasks'])

texts = [
    "I am very happy with your product.",
    "I am not happy with your product.",
    "Your product works as expected.",
]

for text in texts:
    sentiment, confidence = sentiment_analysis(text)
    percentage = round(confidence * 100, 2)
    print(f"{text}")
    print(f"-> {sentiment} (Confidence: {percentage}%)\n")
