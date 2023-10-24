import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from generic_tasks import complete

text = "In this course, we will teach you how to"

completed_text = complete(text, max_length=500)

print(f"Original text: {text}")
print(f"Completed text: {completed_text}")
