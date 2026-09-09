---
pretty_name: Sinhala Diachronic POS & Lemma Corpus
language:
- si
license: other
task_categories:
- token-classification
tags:
- diachronic
- historical-linguistics
- sinhala
- pos-tagging
- lemmatization
- semantic-change
- conllu
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/sinhala_diachronic_pos_lemma.csv
---

# Sinhala Diachronic POS & Lemma Corpus

Token-level part-of-speech tags and lemmas for a corpus of 59 OCR'd historical
Sinhala documents (5th–20th century), built as part of a study on diachronic
semantic change in Sinhala (see **Paper** below). Provided in two formats:

- **`data/sinhala_diachronic_pos_lemma.conllu`** — [CoNLL-U Plus](https://universaldependencies.org/ext-format.html),
  the standard format for POS/lemma-annotated corpora in this field (loadable
  with `conllu`, spaCy, Stanza, UDPipe, etc.).
- **`data/sinhala_diachronic_pos_lemma.csv`** — the same data as a flat CSV,
  for anyone who'd rather not parse CoNLL-U.

## Paper

**Dynamics of meaning: Towards the Evaluation of Diachronic Semantic Change in Sinhala**
Nevidu Jayatilleke, Nisansa de Silva (2026)

📄 [arXiv:2609.08609](https://arxiv.org/abs/2609.08609)

## Loading the dataset

**CSV, via 🤗 `datasets`:**

```python
from datasets import load_dataset

ds = load_dataset("Nevidu/SiDiaC-v.2.5")
print(ds)
print(ds["train"][0])
```

**CSV, via `pandas`** (if you just want a DataFrame):

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Nevidu/SiDiaC-v.2.5",
    repo_type="dataset",
    filename="data/sinhala_diachronic_pos_lemma.csv",
)
df = pd.read_csv(path)
```

**CoNLL-U Plus** (`pip install conllu`):

```python
from huggingface_hub import hf_hub_download
from conllu import parse_incr

path = hf_hub_download(
    repo_id="Nevidu/SiDiaC-v.2.5",
    repo_type="dataset",
    filename="data/sinhala_diachronic_pos_lemma.conllu",
)

with open(path, encoding="utf-8") as f:
    for sentence in parse_incr(f):
        print(sentence.metadata["sent_id"], sentence.metadata["text"])
        for token in sentence:
            print(token["form"], token["lemma"], token["upos"], token["xpos"])
        break  # remove to iterate the whole corpus
```

## Dataset structure

One row/token line per token, grouped into sentences. Columns (CSV) /
per-token fields (CoNLL-U, via `# global.columns`):

| column | description |
|---|---|
| `century` | Century the source document was written in (5, 12, 13, 14, 15, 16, 18, 19, 20 — no attested documents in the 17th century) |
| `sentence_index` | Index of the sentence within its century's document collection |
| `token_index` | Position of the token within its sentence (0-indexed) |
| `word` | Surface form (FORM) |
| `lemma` | Finalized lemma (see "How POS/lemma were finalized" below) |
| `pos` | Finalized POS tag, SinMorphy's own category labels (this dataset's XPOS) |
| `upos` | `pos` mapped to the [Universal POS tagset](https://universaldependencies.org/u/pos/) |
| `sinmorphy_pos`, `sinmorphy_lemma`, `sinmorphy_morphology` | SinMorphy's own single-best pos/lemma/morphology tag (e.g. `කාම+N+NUTR+PL+NOM`) before ensemble resolution |
| `sinmorphy_all_tags`, `sinmorphy_all_morphologies` | SinMorphy's full candidate list when ambiguous (`\|`-separated) |
| `sinling_pos`, `tnt_pos` | The other two taggers' individual votes |
| `doc_id`, `doc_title`, `doc_title_en`, `doc_author`, `doc_genre` | Source document metadata |
| `written_date_start`, `written_date_end`, `issued_date`, `ocr_confidence` | Dating and OCR quality metadata |

CoNLL-U sentence blocks additionally carry `# sent_id`, `# text`, `# century`,
`# doc_id`, `# doc_title`, `# doc_author`, `# written_date` comments.
`FEATS`/`HEAD`/`DEPREL`/`DEPS`/`MISC` are unspecified (`_`) — no morphological
feature mapping or dependency parse is available in this pipeline.

Row counts per century:

| century | rows |
|---|---|
| 5 | 565 |
| 12 | 866 |
| 13 | 13,617 |
| 14 | 5,210 |
| 15 | 8,483 |
| 16 | 1,369 |
| 18 | 4,707 |
| 19 | 12,701 |
| 20 | 17,295 |

**Total: 64,813 tokens, 4,394 sentences, 59 documents.**

`upos` distribution: NOUN 31,890 · VERB 9,963 · PART 9,329 · X 9,160 · PROPN
2,264 · ADJ 1,661 · PRON 546. `doc_genre` spans 8 categories (Non-Fiction;
Religious, Fiction; Poetry, Non-Fiction; Language, Fiction; Religious,
Non-Fiction; History, Non-Fiction; Poetry, Non-Fiction; Medical,
Non-Fiction).

## How `pos` and `lemma` were finalized

Three independent taggers were run over the same tokenized corpus:

1. **SinMorphy** — a Sinhala morphological analyzer (web tool), scraped via
   Selenium (`Morphological Analysis/sinmorphy_automation_v6.ipynb`). Returns
   one or more candidate `(pos, morphology)` pairs per word.
2. **sinling** — a Sinhala NLP library's POS tagger
   (`POS_Tagging/sinling_pos_tagger.ipynb`).
3. **TnT** — a beam-search variant of the Trigrams'n'Tags statistical POS
   tagger, using a pretrained Sinhala model
   (`POS_Tagging/TnT_Tagger_Robust_Alignment.ipynb`).

For words where SinMorphy returned multiple candidates, the candidate is
chosen by priority vote: prefer the one matching TnT's tag, else the one
matching sinling's tag, else the first candidate. `lemma` is then the stem
extracted from the chosen candidate's morphology string (text before the
first `+`). All three taggers' individual outputs are kept in the dataset
(see columns above) so this resolution can be audited or redone.

`upos` is a fixed, non-destructive mapping from the finalized `pos`:
NOUN→NOUN, PROPER NOUN→PROPN, Verb→VERB, PART→PART, ADJECTIVE→ADJ,
PRONOUN→PRON, Proper noun or Invalid→X, FOREIGN WORD→X.

Reproducible end-to-end in
[`build_dataset.ipynb`](build_dataset.ipynb).

## How document metadata was attached

`Data/century_tokens.json` (upstream) concatenates each document's sentences
into a per-century list without preserving document boundaries. Document
provenance is recovered by re-tokenizing each source document (same function
as `Data Processing/Data_Partitioning.ipynb`) and using each document's own
sentence count as a contiguous offset range within its century — documents
are concatenated in a fixed traversal order and each contributes exactly one
uninterrupted block, so content doesn't need to match exactly for this to be
correct, only the sentence *counts* per document.

**Caveat:** the current `.txt` files in `Data/OCR_Final/` do not
byte-for-byte reproduce `century_tokens.json` for every document — a small
number of sentences differ by a token (both files entered the repo in the
same commit and neither has been modified since, so this is a pre-existing
snapshot mismatch, not introduced by this notebook). This doesn't affect
`doc_id`/provenance assignment (which relies only on sentence counts, not
content), but it means a handful of `word`/`lemma` values in this dataset may
differ by a token or two from what you'd get re-tokenizing today's
`OCR_Final/` files directly. Of 59 documents, 4 show a first/last-sentence
mismatch against the frozen corpus (flagged in `build_dataset.ipynb` as
"anchor drift" — informational only; total sentence-count accounting for
every century balances exactly, so document attribution itself is unaffected).

## Known limitations

- **~14.1% of tokens (9,143 rows)** have no SinMorphy analysis at all
  (unrecognized/OOV words, mostly proper nouns) — `pos` is
  `Proper noun or Invalid` (`upos` = `X`) and `lemma` falls back to the raw
  `word`.
- **0.3% of tokens (203 rows)** hit a SinMorphy morphology-string edge case
  with no lemma stem before the first `+` — `lemma` also falls back to the
  raw `word` for these.
- POS/lemma quality depends on the accuracy of the three underlying taggers,
  which have not been independently evaluated against a gold-standard
  Sinhala treebank as part of this pipeline.
- `FEATS` (case, number, tense, etc.) is not populated — SinMorphy's raw
  morphology tag is preserved as-is in `sinmorphy_morphology`/`SinMorphyMorphology`
  instead of being mapped to UD's `Feature=Value` syntax.
- License is unspecified — set the `license` field above (and this notice)
  before publishing to the Hub.

## Provenance

Source texts: `Data/OCR_Final/` (documents with per-document metadata:
title, author, genre, written/issued date, OCR confidence).

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
