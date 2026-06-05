# Sexism Detection

Two-part project on automatic **sexism detection in social-media text**, developed for the
*Natural Language Processing* course (Master of Artificial Intelligence, Year 2 — University of Bologna).

The repository bundles the two course assignments:

- **Assignment 1 — BiLSTM & Transformer:** GloVe + BiLSTM models and a fine-tuned Twitter-RoBERTa transformer.
  Notebook: [`sexism_detection_transformer_bilstm.ipynb`](sexism_detection_transformer_bilstm.ipynb) ·
  Report: [`sexism_detection_transformer_bilstm_report.pdf`](sexism_detection_transformer_bilstm_report.pdf)
- **Assignment 2 — LLM Prompting:** Zero-shot and few-shot prompting of instruction-tuned LLMs.
  Notebook: [`sexism_detection_llm.ipynb`](sexism_detection_llm.ipynb) ·
  Report: [`sexism_detection_llm_report.pdf`](sexism_detection_llm_report.pdf)

**Authors:** Marco Borghi, Arash Foroozanfar, Armina Sadeghi, Razieh Soleimanbeigi

---

## Assignment 1 — BiLSTM & Transformer

**Goal.** Multi-class classification of the *intent* of sexist English tweets from the
[EXIST](http://nlp.uned.es/exist2023/) dataset into four categories:

```
{ '-': 0, 'DIRECT': 1, 'JUDGEMENTAL': 2, 'REPORTED': 3 }
```

**Pipeline (notebook tasks).**
1. **Corpus** — load the three EXIST JSON splits, aggregate the per-annotator labels by
   **majority voting** (dropping items with no clear majority), keep only English (`lang == 'en'`) rows.
2. **Data cleaning** — strip emojis, hashtags, mentions, URLs, special characters and curly quotes; lemmatize with NLTK.
3. **Text encoding** — build a vocabulary as the union of training tokens and **GloVe** (`glove-wiki-gigaword`);
   OOV training tokens get random-normal embeddings, an `<UNK>` token holds the mean embedding, `<PAD>` is a zero vector.
4. **Models**
   - **Baseline BiLSTM** — single bidirectional LSTM → softmax.
   - **Stacked BiLSTM** — two stacked bidirectional LSTM layers → softmax.
   - **Twitter-RoBERTa** — fine-tuned [`cardiffnlp/twitter-roberta-base-hate`](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate) via the HF `Trainer`.
5. **Training & evaluation** — grid-search hyperparameters, train over **3 seeds** (`42, 123, 2024`),
   report mean ± std macro-F1 / precision / recall on validation, pick the best config, evaluate on the test set.
   Class weighting is used to counter label imbalance.
6. **Error analysis** — confusion matrices and top-loss inspection across all three architectures.

**Test-set results (macro-F1).**

| Model | Accuracy | Macro-F1 |
|-------|---------:|---------:|
| Baseline BiLSTM | 0.61 | 0.36 |
| Stacked BiLSTM | 0.69 | 0.47 |
| Twitter-RoBERTa | 0.76 | 0.53 |

---

## Assignment 2 — LLM Prompting

**Goal.** 5-class sexism-type classification of English text via **prompting** (no fine-tuning) on a
**balanced** test set of 300 samples (60 per class):

```
not-sexist, threats, derogation, animosity, prejudiced
```

**Pipeline (notebook sections).**
1. **Model setup** — load two small instruction-tuned LLMs with device-aware optimization
   (4-bit quantization via `bitsandbytes` on CUDA, fp16 on MPS, fp32 on CPU):
   - [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (*Phi3Mini*)
   - [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) (*Qwen3*)
2. **Prompt setup** — chat-template prompts for **zero-shot** and **few-shot** settings; few-shot draws
   balanced demonstrations (`num_per_class = 3`) from `demonstrations.csv`.
3. **Inference** — batched generation, then robust parsing of free-text responses into label IDs
   (returns `-1` when unparseable).
4. **Metrics** — accuracy, macro-F1, and **fail ratio** (share of unparseable outputs).
5. **Error analysis** — performance table, confusion matrices, and NLL-of-true-label ranking to surface
   *confidently wrong* predictions.

**Test-set results.**

| Model | Setting | Accuracy | Macro-F1 | Fail Ratio |
|-------|---------|---------:|---------:|-----------:|
| Phi3Mini | zero-shot | **0.42** | **0.39** | 0.00 |
| Phi3Mini | few-shot | 0.36 | 0.35 | 0.00 |
| Qwen3 | zero-shot | 0.29 | 0.21 | 0.00 |
| Qwen3 | few-shot | 0.24 | 0.14 | 0.00 |

A naive baseline on the balanced 5-class set is ~0.20 accuracy. Phi3Mini zero-shot is the best run;
few-shot demonstrations hurt both models. Fail ratio is 0.00 everywhere — errors are semantic, not formatting.

---

## Setup

Both notebooks were developed for **Google Colab** (GPU runtime), but run locally too.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `torch`, `tensorflow`, `transformers`, `datasets`, `evaluate`, `gensim`,
`nltk`, `scikit-learn`, `bitsandbytes`, `accelerate`, `huggingface_hub`.

**Notes when running locally**
- The LLM notebook requires a Hugging Face login (`hf auth login`) and gated-model access; a GPU is
  strongly recommended for the quantized models.
- The data paths in the notebooks reflect their original Colab/Drive layout
  (e.g. `/content/data/...` and `data/a2_test.csv`). Point them at the folders in this repo
  (`data/data_transformer_bilstm/` and `data/data_llm/`) before running.

---

## Credits / use

Academic coursework. The assignments were designed and provided by
**Federico Ruggeri, Eleonora Mancini, and Paolo Torroni**.

Datasets retain their original licenses (EXIST; the *Explainable Detection of
Online Sexism* data used for Assignment 2).
