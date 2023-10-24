import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from generic_tasks import classify

texts = [
    "This is a course about the Transformers library.",
    "The elections in the United States are over.",
    "The product works as expected.",
]
labels = ["education", "politics", "business"]

for text in texts:
    classification, confidence = classify(text, labels)
    percentage = round(confidence * 100, 2)
    print(f"{text}")
    print(f"-> {classification} (Confidence: {percentage}%)\n")
