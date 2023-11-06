import time

import requests
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def translate(text, from_language=None, to_language=None, **kwargs):
    """
    Generic Translation

    :param text: The text to be translated
    :param from_language: The language of the original text
    :param to_language: The language to translate to
    :param **kwargs:
    :return: The translated text
    """
    known_languages = {
        "en": "English",
        "de": "German",
    }
    if from_language is None:
        from_language = "en"

    if to_language is None:
        to_language = "en"

    if from_language == to_language:
        return text

    # Map language codes to language names
    if from_language in known_languages:
        from_language = known_languages[from_language]

    if to_language in known_languages:
        to_language = known_languages[to_language]

    if to_language == "German":
        to_language = "formal German"

    model = kwargs.get("model", "open-orca-platypus2")
    prompt = f"Translate the following text from {from_language} to {to_language}:\n{text}"

    ollama_url = "http://localhost:11434/api/generate"
    request = {
        "model": model,
        "system": "You are a translator en-de and vice versa. "
                  "DO NOT ADD OR REMOVE INFORMATION. DO NOT ADD COMMENTS. "
                  "If text cannot be translated due to the presence of special characters that are not translatable, simply copy the special characters.",
        "prompt": prompt,
        "options": {
            "temperature": 0.0,
        },
        "stream": False,
    }

    # Send the request to the Ollama URL
    response = requests.post(ollama_url, json=request)

    # Get the response as JSON
    response = response.json()

    # Get the answer from the response
    answer = response["response"]

    return answer

if __name__ == "__main__":
    models = [
        # "mistral", # 87.87s, somewhat bumpy, but otherwise good translation
        # "llama2", # 68.49s, somewhat bumpy, but otherwise good translation
        # "codellama",
        # "vicuna", # 71.57s, quite bumpy, but otherwise good translation
        # "orca-mini", # 72.75s, somewhat bumpy, but otherwise good translation
        "llama2-uncensored", # 71.33s, better, but still not perfect
        # "wizard-vicuna-uncensored", # 71.80s, just like vicuna
        # "nous-hermes", # 69.59s, quite bumpy translation, informal language
        # "phind-codellama",
        "mistral-openorca", # 73.31s, just small failures
        # "wizardcoder",
        "wizard-math", # s71.02s, quite good translation
        # "llama2-chinese", # 74.32s, quite bumpy translation
        # "stable-beluga", # 70.31, just like vicuna
        # "codeup",
        "everythinglm", # 71.68s, good translation, mix of formal and informal language
        # "medllama2",
        # "wizardlm-uncensored", # 75.57s, quite bumpy translation
        # "zephyr", # 77.35s, quite bumpy translation
        # "falcon", # 69.30s, grammatical errors
        # "wizard-vicuna", # 70.45s, quite bumpy translation
        "open-orca-platypus2", # 69.08s, somewhat bumpy, but otherwise good translation
        # "starcoder",
        # "samantha-mistral", # 70.35s, quite bumpy translation
        # "wizardlm", # 75.29s, quite bumpy translation
        # "sqlcoder",
        # "dolphin2.1-mistral", # 70.90s, quite bumpy translation, informal language
        # "nexusraven", # 75.07s, quite bumpy translation
        # "openhermes2-mistral", # 71.98s, bumpy translation
        # "dolphin2.2-mistral", # 67.45s, bumpy translation
        # "codebooga",
    ]

    text_en = """
Dear [Customer Name],

I understand your frustration with the issues you're experiencing in your home.
Water damage and a front door that doesn't close properly can be very disruptive
and stressful. I want to assure you that we take these concerns seriously and
will do everything we can to help resolve them.

Can you please provide more details about the water damage? Is it due to a
recent event or has it been an ongoing issue? And what is the nature of the
problem with the front door? Is it difficult to open or close, or does it not
align properly? Knowing the specifics will help us better understand the
situation and find the best solution.

Once we have more information, we can work together to find a resolution. We may
need to send someone to inspect the issue in person, but we will do our best to
address your concerns as quickly and efficiently as possible.

Thank you for bringing this to our attention, and please let me know if there's
anything else I can help with.

Best regards, [Your Name]
    """

    text_de = """
Sehr geehrter [Kundenname],

ich verstehe Ihre Frustration über die Probleme, die Sie in Ihrem Zuhause
erleben. Wasserschäden und eine Haustür, die nicht richtig schließt, können sehr
störend und stressig sein. Ich möchte Ihnen versichern, dass wir diese Bedenken
ernst nehmen und alles tun werden, um sie zu lösen.

Können Sie bitte weitere Details zu den Wasserschäden angeben? Liegt es an einem
kürzlichen Ereignis oder ist es ein anhaltendes Problem? Und was ist die
Ursache des Problems mit der Haustür? Ist sie schwierig zu öffnen oder zu
schließen, oder ist sie nicht richtig ausgerichtet? Wenn wir die Details kennen,
können wir die Situation besser verstehen und die beste Lösung finden.

Sobald wir mehr Informationen haben, können wir gemeinsam eine Lösung finden.
Wir müssen möglicherweise jemanden schicken, um das Problem persönlich zu
untersuchen, aber wir werden unser Bestes tun, um Ihre Bedenken so schnell und
effizient wie möglich zu beheben.

Vielen Dank, dass Sie uns darauf aufmerksam gemacht haben, und lassen Sie es
mich bitte wissen, wenn ich Ihnen noch weiterhelfen kann.

Mit freundlichen Grüßen, [Ihr Name]
    """

    for model in models:
        print(f"\nModel: {model}")
        try:
            # Start timer
            start_time = time.time()
            translation = translate(text_en, from_language="en", to_language="de")
            # Stop timer
            end_time = time.time()
            duration = end_time - start_time
            print(f"{translation} ({duration:.2f} seconds)")

            # Start timer
            start_time = time.time()
            translation = translate(text_de, from_language="de", to_language="en")
            # Stop timer
            end_time = time.time()
            duration = end_time - start_time
            print(f"{translation} ({duration:.2f} seconds)")
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except:
            print(f"Error in {model}")
            continue