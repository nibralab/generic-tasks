import re

import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


def summarize(long_text: str) -> str:
    """
    Generic Summarizer

    :param long_text: The text to be summarized
    :return: The summary
    """
    model_name = "sshleifer/distilbart-cnn-12-6"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    prediction = summarizer(long_text)

    summary = prediction[0]['summary_text']

    # Remove spaces before punctuation ('.', ',', '?', '!') using RegEx
    summary = re.sub(r'\s([?.!,](?:\s|$))', r'\1', summary)

    return summary
