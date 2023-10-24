import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


def answer(question: str, context: str):
    """
    Generic Question-Answering

    :param question:  The question to be answered
    :param context:   Context containing relevant information
    :return:          The answer (str) and the confidence level (float)
    """
    model_name = "distilbert-base-cased-distilled-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    answerer = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    prediction = answerer(question=question, context=context)

    answer = prediction['answer']
    confidence = prediction['score']

    return answer, confidence
