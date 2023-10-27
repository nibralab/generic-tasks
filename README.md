# Generic Tasks for AI Prototypes

This repository contains a set of generic tasks for AI prototypes.

The package provides the following functions:

- `split_paragraphs()`: Generic function to split text into paragraphs.
- `split_sentences()`: Generic function to split text into sentences.
- `classify()`: Generic text classification.
- `complete()`: Generic text completion.
- `answer()`: Generic question answering.
- `sentiment_analysis()`: Generic sentiment analysis.
- `summarize()`: Generic text summarization.
- `translate()`: Generic text translation.

All functions use the first CUDA GPU, if available.

For usage examples, see the [tests](tests) directory.

## Installation

The following command syntax can be used to install the default branch of the project: 

```bash
$ pip install git+https://github.com/nibralab/generic-tasks.git
```

To install a specific version, enter:

```bash
$ pip install git+https://github.com/nibralab/generic-tasks.git@<tag>
```

where `<tag>` is the tag name of the version to install.

If you want to include the project in a requirements file, use the following syntax:

```text
generic-tasks==<tag>
-e https://github.com/nibralab/generic-tasks.git@<tag>#egg=generic_tasks
```

The package will then be installed with

```bash
$ pip install -r requirements.txt
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).
