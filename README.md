# Local email-classification scoring demo

A small, self-contained Python example that uses a local GGUF language model to score a multiple-choice email-classification prompt.

## Attribution

This project reproduces the example from ["JEV in 25 Lines"](https://www.nobodywho.ai/posts/jev-in-25-lines/) by NobodyWho. Credit for the original example and its approach belongs to that post.

The script loads [`Qwen/Qwen3-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) through `llama-cpp-python`, asks the model to choose among `Legitimate`, `Spam`, and `Phishing`, and calculates a probability for each choice from the next-token logits. It currently evaluates one hard-coded email:

> Payroll asks for your password on a non-company sign-in page.

## How it works

`main.py`:

1. Downloads or reuses the `Qwen3-0.6B-Q8_0.gguf` model from Hugging Face.
2. Runs the prompt through the model without generating a response.
3. Reads the logits for the `A`, `B`, and `C` answer tokens.
4. Applies log-softmax normalization and prints logits, log probabilities, and probabilities for the three labels.

## Requirements

- Python 3.12 or later
- Sufficient disk space and memory for the Q8 GGUF model; the first run downloads the model and may take a while.

The direct dependencies are listed in `requirements.txt` and in the PEP 723 metadata at the top of `main.py`:

- `huggingface-hub`
- `llama-cpp-python`
- `numpy`

## Run

With [uv](https://docs.astral.sh/uv/), the script metadata is enough:

```bash
uv run main.py
```

Or use the included virtual environment setup:

```bash
source .venv/bin/activate
python main.py
```

To create that environment from scratch:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

## Customize the example

Edit `email` in `main.py` to score a different message. You can also change `choices`, `labels`, or the Hugging Face `repo_id` and `filename` to use another compatible GGUF model.

## License

[MIT](LICENSE)
