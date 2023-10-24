import re
import nltk.data


def split_sentences(text: str, language: str='en'):
    """
    Generic function to split text into sentences.

    :param text: The text to be split
    :param language:  The language of the text
    :return:  List of sentences
    """
    if language == 'de':
        pickle = 'tokenizers/punkt/PY3/german.pickle'
    elif language == 'en':
        pickle = 'tokenizers/punkt/PY3/english.pickle'
    else:
        raise ValueError(f'Language {language} not supported.')

    tokenizer = nltk.data.load(pickle)
    sentences = tokenizer.tokenize(text.strip())
    sentences = [re.sub(r'\s+', ' ', sentence) for sentence in sentences]

    return sentences
