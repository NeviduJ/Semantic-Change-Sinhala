# Diachronic Semantic Change in Sinhala

A study on **diachronic semantic change in Sinhala**: how word meanings have shifted
across 6 centuries (13th, 14th, 15th, 18th, 19th, 20th) of Sinhala textual history,
using both static (Word2Vec, FastText) and contextual (XLM-R, SinLlama, fine-tuned Llama-3.1-8B) word
embeddings, alongside a from-scratch Dynamic Bernoulli Embeddings implementation. This
repository contains the full pipeline behind the paper below, including the
**SiDiaC-v.2.5** POS/lemma dataset it produces.

Nevidu Jayatilleke, Nisansa de Silva (2026)

📄 [arXiv:2609.08609](https://arxiv.org/abs/2609.08609)

## Overview

The pipeline covers, in order:

1. **Corpus construction** (`Data Processing/`, `Data/OCR_Final/`) — 186 OCR'd historical
   Sinhala documents, each with metadata (title, author, genre, written/issued date, OCR
   confidence), partitioned by century of composition and tokenized into per-century
   sentence collections.
2. **Morphology & POS tagging** (`Morphological Analysis/`, `POS_Tagging/`) — an
   ensemble of three taggers (SinMorphy, sinling, a beam-search TnT tagger) reconciled
   into finalized POS tags and lemmas.
3. **Static embeddings** (`Word2Vec/`, `Fasttext/`) — one model trained per century,
   compared across time via three interchangeable alignment methods: classical
   **Orthogonal Procrustes**, a learned **Neural-OP** transform, and a rotation-free
   **Similarity Matrix Alignment** (anchor-profile correlation).
4. **Contextual embeddings** (`Contextualised embeddings/`) — per-token vectors from
   XLM-R, SinLlama, and a fine-tuned Llama-3.1-8B, aggregated into per-century lemma
   centroids. A leave-one-out (LOO) analysis decomposes each century-pair's semantic
   drift into a *persistent shift* (stable core meaning) versus a small subset of
   *high-impact contextual usages* driving the observed change.
5. **Dynamic Bernoulli Embeddings** (`Dynamic_Bernoulli_Embeddings/`) — a from-scratch
   reimplementation of Rudolph & Blei's probabilistic time-series embedding model,
   jointly fitting a single generative model across all centuries with a Gaussian
   random-walk prior linking per-century embeddings.

## Repository structure

```
Data/                           Corpus, drift-analysis outputs, embeddings, reports
Data Processing/                Corpus partitioning, lemma reconstruction, drift/dip analysis
Morphological Analysis/         SinMorphy automation
POS_Tagging/                    sinling, TnT, and merged POS/lemma finalization
Word2Vec/, Fasttext/            Per-century training + three alignment methods each
Contextualised embeddings/      XLM-R / SinLlama / Llama-FT centroid + LOO drift analysis
Dynamic_Bernoulli_Embeddings/   Dynamic Bernoulli Embeddings (gitignored — large fit artifacts)
Sinhala-Diachronic-POS-Lemma-Dataset/  Shareable, field-conforming POS/lemma dataset build
Paper/                          The paper PDF
```

## Datasets

The finalized, field-conforming (CoNLL-U Plus + rich CSV) POS/lemma corpus built from
this pipeline is published separately on the Hugging Face Hub as
[`Nevidu/SiDiaC-v.2.5`](https://huggingface.co/datasets/Nevidu/SiDiaC-v.2.5) (private) —
see `Sinhala-Diachronic-POS-Lemma-Dataset/` for the build notebook.

## Environment

- The default `.venv` runs Python 3.14, on which `gensim`'s Cython extensions currently
  fail to build. Use `.venv_2` (Python 3.12) for anything touching Word2Vec/FastText
  models — it has `gensim`, `torch`, `kneed`, `scipy`, `scikit-learn`, and `matplotlib`
  installed.
- Large binary artifacts (`Data/*.pt*`, `*.model`, `*.npy`, etc.) are gitignored; one
  large CSV (`Data/LlaMa_FT_lemma_drift_analysis_upd.csv`, ~220MB) is tracked via
  [Git LFS](https://git-lfs.github.com).

## Citation

```bibtex
@article{jayatilleke2026dynamicsmeaningevaluationdiachronic,
      title={Dynamics of meaning: Towards the Evaluation of Diachronic Semantic Change in Sinhala}, 
      author={Nevidu Jayatilleke and Nisansa de Silva},
      year={2026},
      eprint={2609.08609},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.08609}, 
}
```
