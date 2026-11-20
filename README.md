[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/u99pmfr2sc/badge.svg)](https://msrit.ask.thili.ai/b/u99pmfr2sc)
# Sports Commentary Generator

Type a simulated match event into Streamlit to generate a short call. For example, **Tara hits a three-pointer to level it at 81-81.** produces a basketball call with Tara's name and the supplied score.

## Run on Windows with Python 3.12

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app/main.py
```

Open the local URL printed in the terminal. First generation downloads the public GPT-2 checkpoint (about 550 MB) and tokenizer. Later runs use the cache. No account or API key is required. The trained rank-eight adapter is included.

## How it works

1. Generate 48 original football, basketball and cricket event/call pairs. Reserve 12 pairs with unseen players for evaluation.
2. Run frozen GPT-2 small on the event and its target continuation. Predict only commentary tokens. Retain the language-model head columns corresponding to the commentary vocabulary.
3. Inject two trainable matrices into that output projection. Train the low-rank update using cross-entropy and Adam, keeping GPT-2 frozen. The NumPy implementation uses the public checkpoint and tokenizer.
4. On load, merge the update once as W + (alpha/r)AB, with rank 8 and alpha 16. Inference uses the merged projection.
5. For six recognized play descriptions, score two grounded calls by mean token log probability and copy any explicit score. Other descriptions use greedy generation from the learned vocabulary. Open-ended events can produce repetitive or inaccurate calls.

## Reproduce

Run in the environment created above:

```powershell
.\.venv\Scripts\python -m app.train
.\.venv\Scripts\python -m app.evaluate
.\.venv\Scripts\python -m pytest -q
```

The evaluation compares a matching call against a mismatched call about the same player, using teacher-forced model scores before and after adaptation. Base GPT-2 preferred the matched call on 8/12 pairs. The adapted model preferred it on 12/12. The evaluation also prints each generated call for inspection. These held-out cases reuse the training event types and test transfer to new names.

## Files

`app/data.py` defines fixtures. `app/base.py` loads GPT-2 and runs its transformer layers. `app/adapter.py` implements low-rank training. `app/train.py` produces `app/sports_adapter.npz`. `app/generator.py` merges weights and decodes commentary. `app/main.py` runs the interface.

Base weights and tokenizer: https://huggingface.co/gpt2

