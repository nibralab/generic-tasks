import re

def split_paragraphs(text: str):
    """
    Generic function to split text into paragraphs.

    :param text: The text to be split
    :return: List of paragraphs
    """
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [paragraph.strip() for paragraph in paragraphs]

    return paragraphs
