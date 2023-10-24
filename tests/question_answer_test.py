import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from generic_tasks import answer

question = "Where do I work?"
context = "My name is Sylvain and I work at Hugging Face in Brooklyn"

answer, confidence = answer(question, context)
percentage = round(confidence * 100, 2)

print(f"Question: {question}")
print(f"Answer: {answer} (Confidence: {percentage}%)")
