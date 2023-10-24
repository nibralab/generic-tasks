"""
This is the Generic Tasks package.

The package provides the following functions:

- split_paragraphs: Generic function to split text into paragraphs.
- split_sentences: Generic function to split text into sentences.
- classify: Generic text classification.
- complete: Generic text completion.
- answer: Generic question answering.
- sentiment_analysis: Generic sentiment analysis.
- summarize: Generic text summarization.
- translate: Generic text translation.

"""

__version__ = "0.1.0"
__author__ = 'Niels Braczek, AI-Schmiede'

from .paragraphs import split_paragraphs
from .sentences import split_sentences

from .classification import classify
from .completion import complete
from .question_answer import answer
from .sentiment import sentiment_analysis
from .summarization import summarize
from .translation import translate
