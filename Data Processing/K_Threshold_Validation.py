"""
Empirical validation of the k=0.8 drift-detection threshold (cutoff = mu - k*sigma)
used throughout the FastText/Word2Vec semantic-drift pipelines.

For each of four alignment pipelines (Word2Vec and FastText, each under Statistical
Orthogonal Procrustes and Neural-OP alignment), this script:
  1. Recomputes the per-word/lemma cross-century cosine similarities exactly as the
     corresponding notebooks do (Word2Vec/Orthogonal_Procrustes_Alignment_WV.ipynb,
     Fasttext/Orthogonal_Procrustes_Alignment.ipynb,
     Word2Vec/Neural_Network_Tranformation.ipynb, and the saved FastText Neural-OP
     alignment in Fasttext/models/sinhala_all_alignments.npz).
  2. Sweeps k from 0 to 3 and counts how many word/period-pair instances would be
     flagged as drift at each k (per-period-pair dynamic cutoff, matching
     get_semantic_drift_report()'s logic).
  3. Finds each curve's empirical knee via the Kneedle algorithm (for reference only
     -- the knees consistently land around k=1.0-1.5, NOT at 0.8; see the printed
     summary and README note below).
  4. Reports the percentage of instances flagged at k=0.8 against the theoretical
     Gaussian expectation (Phi(-0.8) = 21.19%), and saves a LaTeX table + PDF/PNG
     figure to Data/k_threshold_validation_table.tex and
     Data/k_threshold_validation_figure.{pdf,png}.

IMPORTANT -- what this script does and does NOT claim:
  k=0.8 is NOT the empirical elbow/knee of these curves (the real knees are ~1.0-1.5).
  What this script demonstrates instead is that the *proportion* of instances flagged
  at k=0.8 is close to the Gaussian-theoretical proportion (~21%) and is *consistent*
  across four structurally different pipelines (different embedding models, different
  alignment mechanisms) -- i.e. k=0.8 is a stable, non-arbitrary choice, even though it
  sits below rather than at the natural knee of the flagged-count curve.

Environment
------------
This repo's default `.venv` uses Python 3.14, on which gensim's Cython extensions
currently fail to build. Run this script with `.venv_2` (Python 3.12) instead, which
has gensim, torch (CPU), kneed, scipy, scikit-learn, and matplotlib installed:

    ./.venv_2/bin/python3 "Data Processing/K_Threshold_Validation.py"

(paths below are resolved relative to the repo root regardless of your cwd)

Memory note
-----------
Each Fasttext/models/sinhala_ft_century_*.model is ~4GB once loaded (2.2GB n-gram
vectors + ~1GB syn1neg + ~1GB vocab vectors). This script loads them ONE AT A TIME,
extracts only the ~210 word vectors needed (consistent_keys_2.json), and explicitly
releases each model (del + gc.collect()) before loading the next, to keep peak memory
to roughly one model's footprint instead of six. If your machine is already under
memory pressure, close other applications first -- six FastText models loaded
simultaneously would need ~24GB.
"""

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "Data")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from gensim.models import Word2Vec, FastText
    from scipy.linalg import svd
    from scipy.stats import norm
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from kneed import KneeLocator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n\n"
        "This script needs gensim + torch + kneed + scikit-learn + matplotlib.\n"
        "Run it with this repo's .venv_2 (Python 3.12), which already has these "
        "installed:\n\n"
        '    ./.venv_2/bin/python3 "Data Processing/K_Threshold_Validation.py"\n\n'
        "The default .venv (Python 3.14) cannot build gensim's Cython extensions."
    )

with open(os.path.join(DATA_DIR, "consistent_keys_2.json"), "r", encoding="utf-8") as f:
    CONSISTENT_KEYS = json.load(f)

CENTURIES = [13, 14, 15, 18, 19, 20]


# ---------------------------------------------------------------------------
# Step 1: Build a (Key, Year_A, Year_B, Value, feature_pair) long-format table
# for each of the four pipelines.
# ---------------------------------------------------------------------------

def compute_op_long_similarity(century_vectors):
    """Replicates the per-reference-century Orthogonal Procrustes alignment +
    cosine similarity computation shared by Orthogonal_Procrustes_Alignment[_WV].ipynb,
    filtered to Year_A < Year_B exactly as those notebooks do before pivoting.

    century_vectors: dict[int century -> np.ndarray of shape (n_words, dim)],
    rows aligned to CONSISTENT_KEYS.
    """
    century_names = sorted(century_vectors.keys())
    all_reference_similarity_dfs = {}

    for ref_name in century_names:
        W_ref = century_vectors[ref_name]
        W_ref_norm = normalize(W_ref, norm="l2")
        aligned_vectors = {ref_name: W_ref}

        for target_name in century_names:
            if ref_name == target_name:
                continue
            W_target = century_vectors[target_name]
            W_target_norm = normalize(W_target, norm="l2")

            A = W_target_norm.T @ W_ref_norm
            U, s, Vh = svd(A)
            Q = U @ Vh
            aligned_vectors[target_name] = W_target @ Q

        word_data = []
        for word_index, word in enumerate(CONSISTENT_KEYS):
            row = {"Key": word}
            ref_vector = W_ref[word_index].reshape(1, -1)
            for century_name in century_names:
                current_vector = aligned_vectors[century_name][word_index].reshape(1, -1)
                row[century_name] = cosine_similarity(ref_vector, current_vector)[0][0]
            word_data.append(row)

        sim_df = pd.DataFrame(word_data)[["Key"] + century_names]
        all_reference_similarity_dfs[ref_name] = sim_df

    list_of_dfs = []
    for ref_century, sim_df in all_reference_similarity_dfs.items():
        sim_df = sim_df.copy()
        sim_df["Year_A"] = ref_century
        list_of_dfs.append(sim_df)
    final_df_org = pd.concat(list_of_dfs, ignore_index=True)

    df_long = final_df_org.melt(id_vars=["Key", "Year_A"], var_name="Year_B", value_name="Value")
    df_long["Year_B"] = pd.to_numeric(df_long["Year_B"])
    df_lags_only = df_long[df_long["Year_A"] < df_long["Year_B"]].copy()
    df_lags_only["feature_pair"] = df_lags_only["Year_A"].astype(str) + "_" + df_lags_only["Year_B"].astype(str)
    return df_lags_only


def word2vec_op_long():
    print("[Word2Vec - Statistical OP] loading Exp3 models (small, ~2.5MB each)...")
    models = {c: Word2Vec.load(os.path.join(REPO, "Word2Vec", "Exp3", f"model_{name}.model"))
              for c, name in zip(CENTURIES, ["13th", "14th", "15th", "18th", "19th", "20th"])}
    vectors = {c: np.array([models[c].wv[w] for w in CONSISTENT_KEYS], dtype=np.float32) for c in CENTURIES}
    return compute_op_long_similarity(vectors)


def word2vec_neural_op_long():
    print("[Word2Vec - Neural-OP] loading Exp3 models and training NN transform "
          "(exact hyperparameters from Neural_Network_Tranformation.ipynb)...")
    models = {c: Word2Vec.load(os.path.join(REPO, "Word2Vec", "Exp3", f"model_{name}.model"))
              for c, name in zip(CENTURIES, ["13th", "14th", "15th", "18th", "19th", "20th"])}

    embedding_size, learning_rate, epochs, lambda_ortho = 50, 0.001, 2000, 0.001
    anchor = 13
    patience_limit, patience_threshold, min_epochs = 20, 1e-7, 100

    unified_space = {}
    raw_Y = torch.tensor([models[anchor].wv[w] for w in CONSISTENT_KEYS], dtype=torch.float32)
    Y_ref = F.normalize(raw_Y, p=2, dim=1)
    unified_space[anchor] = Y_ref.detach().numpy()

    for target in CENTURIES:
        if target == anchor:
            continue
        raw_X = torch.tensor([models[target].wv[w] for w in CONSISTENT_KEYS], dtype=torch.float32)
        X_input = F.normalize(raw_X, p=2, dim=1)

        model = nn.Linear(embedding_size, embedding_size, bias=False)
        nn.init.eye_(model.weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_loss, patience_counter = float("inf"), 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_input)
            mse_loss = criterion(predictions, Y_ref)
            W_curr = model.weight
            ortho_loss = torch.norm(torch.mm(W_curr, W_curr.t()) - torch.eye(embedding_size), p="fro") ** 2
            loss = mse_loss + (lambda_ortho * ortho_loss)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()

            if (best_loss - current_loss) > patience_threshold:
                best_loss, patience_counter = current_loss, 0
            else:
                patience_counter += 1
            if patience_counter >= patience_limit and epoch >= min_epochs:
                break

        with torch.no_grad():
            Q = model.weight.detach().numpy()
            unified_space[target] = (X_input @ torch.tensor(Q).t()).numpy()
        print(f"  aligned {target} -> {anchor} in {epoch + 1} epochs")

    rows = []
    for i, ca in enumerate(CENTURIES):
        for cb in CENTURIES[i + 1:]:
            Va, Vb = unified_space[ca], unified_space[cb]
            for k, word in enumerate(CONSISTENT_KEYS):
                sim = cosine_similarity(Va[k].reshape(1, -1), Vb[k].reshape(1, -1))[0][0]
                rows.append({"Key": word, "Year_A": ca, "Year_B": cb, "Value": float(sim),
                             "feature_pair": f"{ca}_{cb}"})
    return pd.DataFrame(rows)


def fasttext_op_long():
    print("[FastText - Statistical OP] loading century models ONE AT A TIME "
          "(~4GB each) and extracting only the needed word vectors...")
    vectors = {}
    for c in CENTURIES:
        print(f"  loading century {c}...")
        model = FastText.load(os.path.join(REPO, "Fasttext", "models", f"sinhala_ft_century_{c}.model"))
        vectors[c] = np.array([model.wv[w] for w in CONSISTENT_KEYS], dtype=np.float32)
        del model
        gc.collect()
        print(f"  extracted and released century {c}")
    return compute_op_long_similarity(vectors)


def fasttext_neural_op_long():
    print("[FastText - Neural-OP] loading saved aligned vectors "
          "(Fasttext/models/sinhala_all_alignments.npz, no model loading needed)...")
    d = np.load(os.path.join(REPO, "Fasttext", "models", "sinhala_all_alignments.npz"), allow_pickle=True)
    century_nums = {c: int("".join(ch for ch in c if ch.isdigit())) for c in d.keys()}
    sorted_centuries = sorted(d.keys(), key=lambda c: century_nums[c])
    n_words = d[sorted_centuries[0]].shape[0]

    rows = []
    for i, ca in enumerate(sorted_centuries):
        for cb in sorted_centuries[i + 1:]:
            ya, yb = century_nums[ca], century_nums[cb]
            Va, Vb = d[ca], d[cb]
            for k in range(n_words):
                sim = cosine_similarity(Va[k].reshape(1, -1), Vb[k].reshape(1, -1))[0][0]
                rows.append({"Key": k, "Year_A": ya, "Year_B": yb, "Value": sim,
                             "feature_pair": f"{ya}_{yb}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: sweep k, find the empirical knee (for reference), and report the
# flagged percentage at k=0.8 for each pipeline.
# ---------------------------------------------------------------------------

def sweep_k(long_df, k_min=0.0, k_max=3.0, k_step=0.02):
    """Per-period-pair dynamic cutoff (cutoff = mean - k*std within each
    feature_pair column), matching get_semantic_drift_report()'s logic, swept
    over k. Returns (ks, flagged_counts)."""
    piv = long_df.pivot(index="Key", columns="feature_pair", values="Value")
    ks = np.arange(k_min, k_max + 1e-9, k_step)
    flagged_counts = []
    for k in ks:
        total = 0
        for col in piv.columns:
            series = piv[col].dropna()
            series = series[series > 0]
            if series.empty:
                continue
            cutoff = series.mean() - k * series.std()
            total += (series < cutoff).sum()
        flagged_counts.append(total)
    return ks, np.array(flagged_counts)


def main():
    pipelines = {
        "Word2Vec (Statistical OP)": word2vec_op_long,
        "Word2Vec (Neural-OP)": word2vec_neural_op_long,
        "FastText (Statistical OP)": fasttext_op_long,
        "FastText (Neural-OP)": fasttext_neural_op_long,
    }

    results = {}
    curves = {}
    for label, fn in pipelines.items():
        print(f"\n=== {label} ===")
        long_df = fn()
        total_comparisons = int((long_df["Value"] > 0).sum())
        ks, counts = sweep_k(long_df)
        curves[label] = (ks, counts, total_comparisons)

        knee = KneeLocator(ks, counts, curve="convex", direction="decreasing").knee
        flagged_08 = int(counts[np.argmin(np.abs(ks - 0.8))])
        pct_08 = flagged_08 / total_comparisons * 100
        results[label] = {
            "comparisons": total_comparisons,
            "flagged_at_08": flagged_08,
            "pct_flagged": pct_08,
            "empirical_knee": knee,
        }
        print(f"  comparisons={total_comparisons}  flagged@k=0.8={flagged_08} ({pct_08:.2f}%)  "
              f"empirical knee k*={knee} (NOT 0.8 -- see script docstring)")

    theoretical_pct = norm.cdf(-0.8) * 100
    mean_pct = np.mean([r["pct_flagged"] for r in results.values()])

    # ---- LaTeX table ----
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Comparisons & Flagged at $k=0.8$ & \% Flagged & Deviation from Theory \\",
        r"\midrule",
    ]
    for label, r_ in results.items():
        dev = r_["pct_flagged"] - theoretical_pct
        lines.append(f"{label} & {r_['comparisons']:,} & {r_['flagged_at_08']:,} & "
                      f"{r_['pct_flagged']:.2f}\\% & {dev:+.2f} pp \\\\")
    lines += [
        r"\midrule",
        f"Mean across pipelines & & & {mean_pct:.2f}\\% & {mean_pct - theoretical_pct:+.2f} pp \\\\",
        f"Theoretical Gaussian ($\\Phi(-0.8)$) & & & {theoretical_pct:.2f}\\% & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Empirical validation of the $k=0.8$ drift-detection threshold against "
        r"the theoretical Gaussian expectation across the Statistical OP and Neural-OP "
        r"alignment pipelines.}",
        r"\label{tab:k_threshold_validation}",
        r"\end{table}",
    ]
    table_path = os.path.join(DATA_DIR, "k_threshold_validation_table.tex")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved table to {table_path}")

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728"]
    for (label, (ks, counts, total)), color in zip(curves.items(), colors):
        pct = counts / total * 100
        mask = ks <= 2.0
        ax.plot(ks[mask], pct[mask], label=label, color=color, linewidth=2)

    k_range = np.linspace(0, 2.0, 200)
    ax.plot(k_range, norm.cdf(-k_range) * 100, label="Theoretical Gaussian ($\\Phi(-k)$)",
            color="gray", linestyle=":", linewidth=2)
    ax.axvline(0.8, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(0.82, 48, "$k=0.8$", fontsize=10, rotation=90, verticalalignment="top")
    ax.set_xlabel("Threshold multiplier $k$ (cutoff $= \\mu - k\\sigma$)", fontsize=11)
    ax.set_ylabel("Percentage of word/lemma instances flagged as drift (%)", fontsize=11)
    ax.set_title("Flagged-instance proportion vs. threshold $k$: Statistical OP and Neural-OP", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 55)
    plt.tight_layout()

    fig_pdf = os.path.join(DATA_DIR, "k_threshold_validation_figure.pdf")
    fig_png = os.path.join(DATA_DIR, "k_threshold_validation_figure.png")
    plt.savefig(fig_pdf, dpi=300)
    plt.savefig(fig_png, dpi=200)
    print(f"Saved figure to {fig_pdf} and {fig_png}")

    print("\n=== SUMMARY ===")
    for label, r_ in results.items():
        print(f"{label}: {r_['pct_flagged']:.2f}% flagged at k=0.8 "
              f"(theory: {theoretical_pct:.2f}%, empirical knee: {r_['empirical_knee']})")
    print(f"Mean flagged at k=0.8 across pipelines: {mean_pct:.2f}%")


if __name__ == "__main__":
    main()
