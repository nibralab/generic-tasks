import re
import nltk.data

def crop_bullet(text: str) -> (str, str):
    """
    Crop and remember leading bullet using regex
    """
    bullet_pattern = r"^[ \t]*[\*-][ \t]+"  # Leading bullet

    bullet = ""
    if re.search(bullet_pattern, text):
        bullet = re.search(bullet_pattern, text).group(0)
        text = re.sub(bullet_pattern, "", text)

    return bullet, text

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
    bullet, text = crop_bullet(text)
    sentences = tokenizer.tokenize(text)

    # Prepend the first sentence with the bullet if sentences are not empty
    if sentences:
        sentences[0] = bullet + sentences[0]

    return sentences
