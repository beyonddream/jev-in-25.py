# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub", "llama-cpp-python", "numpy"]
# ///

import numpy
from llama_cpp import Llama

# Really, you can use any GGUF model from https://huggingface.co/models?library=gguf

model = Llama.from_pretrained(
  repo_id="Qwen/Qwen3-0.6B-GGUF",
  filename="Qwen3-0.6B-Q8_0.gguf",
  n_ctx=512,
  logits_all=True,
  verbose=False,
)

labels = ["A", "B", "C"]
choices = ["Legitimate", "Spam", "Phishing"]
email = "Payroll asks for your password on a non-company sign-in page."
options = "\n".join(
  f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
)
prompt = f"""<|im_start|>system
Choose one option.<|im_end|>
<|im_start|>user
Email: {email}\n\n{options}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n"""
model.eval(tokens=model.tokenize(text=prompt.encode(), add_bos=False, special=True))

logits = model.scores[model.n_tokens - 1]
token_ids = [model.tokenize(text=label.encode(), add_bos=False)[0] for label in labels]
choice_logits = numpy.asarray([logits[token_id] for token_id in token_ids])
logprobs = choice_logits - numpy.logaddexp.reduce(choice_logits)
probabilities = numpy.exp(logprobs)

for name, scores in (
  ("Logits", choice_logits),
  ("Log probabilities", logprobs),
  ("Probabilities", probabilities),
):
  values = numpy.round(scores.astype(float), 3).tolist()
  print(f"{name}:", dict(zip(choices, values, strict=True)))
