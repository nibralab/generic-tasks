import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import importlib
import generic_tasks
print(importlib.reload(generic_tasks))

text = "In this course, we will teach you how to"

print("Calling generic_tasks.complete()")
completed_text = generic_tasks.complete(text, max_length=500)

print(f"{text} {completed_text}")
