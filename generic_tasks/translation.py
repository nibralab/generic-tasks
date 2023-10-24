import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def translate(text, from_language=None, to_language=None):
    """
    Generic Translation

    :param text: The text to be translated
    :param from_language: The language of the original text
    :param to_language: The language to translate to
    :return: The translated text
    """
    if from_language is None:
        from_language = "en"

    if to_language is None:
        to_language = "en"

    if from_language == to_language:
        return text

    task = f"translation_{from_language}_to_{to_language}"
    model_name = f"Helsinki-NLP/opus-mt-{from_language}-{to_language}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    translator = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    translation = translator(text)

    return translation[0]['translation_text']
