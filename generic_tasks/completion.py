import torch
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

from generic_tasks import split_sentences


def complete(text: str, max_length: int = 50, num_return_sequences: int = 1):
    """
    Generic Text Completion

    :param text: The text to be completed
    :param max_length:
    :param num_return_sequences:
    :return: The completed text
    """
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    prediction = generator(
        text,
        max_new_tokens=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=50256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        repetition_penalty=1.2,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        early_stopping=True,
        use_cache=True
    )

    generated_text = prediction[0]['generated_text']

    # Remove the original text from the generated text.
    generated_text = generated_text[len(text):]

    generated_text = split_sentences(generated_text)

    # Remove last sentence, if incomplete, i.e., it does not end with '.', '!' or '?'.
    if len(generated_text) > 1 and generated_text[-1][-1] not in ['.', '!', '?']:
        generated_text = generated_text[:-1]

    return generated_text[0]
