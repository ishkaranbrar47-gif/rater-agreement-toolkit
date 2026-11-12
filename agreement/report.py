"""Turn a two-rater table into a full agreement summary, chart and Markdown section."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.contingency_tables import SquareTable  # noqa: E402

from .io import table_to_pairs  # noqa: E402
from .metrics import bootstrap_ci, cohens_kappa, interpret_kappa, krippendorff_alpha  # noqa: E402

INK, ACCENT = "#16202B", "#0E7C66"


def summarize(table, rater_a, rater_b):
    """All statistics for one square table of ordered categories."""
    t = table.astype(float)
    order = list(t.index)
    code = {c: i for i, c in enumerate(order)}
    r1, r2 = table_to_pairs(t)
    a = np.array([code[x] for x in r1])
    b = np.array([code[x] for x in r2])
    n = int(t.values.sum())

    kappas = {}
    for w in (None, "linear", "quadratic"):
        k = cohens_kappa(table=t.values, weights=w)
        lo, hi = bootstrap_ci(a, b, weights=w)
        kappas[w or "unweighted"] = (k, lo, hi)

    diag = np.diag(t.values)
    specific = 2 * diag / (t.sum(axis=1).values + t.sum(axis=0).values)
    off = [
        (order[i], order[j], int(t.values[i, j]))
        for i in range(len(order))
        for j in range(len(order))
        if i != j and t.values[i, j] > 0
    ]
    off.sort(key=lambda x: -x[2])

    # Stuart-Maxwell test: do the two raters use the categories at different rates?
    sm = SquareTable(t.values, shift_zeros=False).homogeneity()
    marg = pd.DataFrame({rater_a: t.sum(axis=1).values / n, rater_b: t.sum(axis=0).values / n}, index=order)

    return {
        "n": n,
        "percent": diag.sum() / n,
        "kappas": kappas,
        "alpha_ordinal": krippendorff_alpha(np.vstack([a, b]).astype(float), "ordinal"),
        "specific": dict(zip(order, specific, strict=True)),
        "top_disagreements": off[:3],
        "stuart_maxwell": (float(sm.statistic), float(sm.pvalue), int(sm.df)),
        "marginals": marg,
    }


def heatmap(table, title, rater_a, rater_b, path):
    t = table.values.astype(float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(t, cmap="Greens")
    ax.set_xticks(range(t.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(t.shape[0]), table.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, int(t[i, j]), ha="center", va="center", color="white" if t[i, j] > t.max() * 0.6 else INK)
    ax.set_xlabel(rater_b)
    ax.set_ylabel(rater_a)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt_p(p):
    return "< .001" if p < 0.001 else f"{p:.3f}"


def markdown_section(name, s, rater_a, rater_b, image):
    k = s["kappas"]
    rows = "\n".join(
        f"| {label} | {v[0]:.3f} | {v[1]:.2f} to {v[2]:.2f} | {interpret_kappa(v[0])} |" for label, v in k.items()
    )
    spec = ", ".join(f"{c} {v:.0%}" for c, v in s["specific"].items())
    top = "; ".join(f"{a} → {b} ({n} items)" for a, b, n in s["top_disagreements"])
    chi, p, dof = s["stuart_maxwell"]
    marg = s["marginals"].map(lambda v: f"{v:.0%}")
    marg_rows = "\n".join(f"| {c} | {r[rater_a]} | {r[rater_b]} |" for c, r in marg.iterrows())
    return f"""## {name}

{s['n']} items rated by {rater_a} and {rater_b}. Raw agreement {s['percent']:.0%}.

| Cohen's kappa | value | 95% bootstrap CI | Landis & Koch |
| --- | --- | --- | --- |
{rows}

Krippendorff's alpha (ordinal): {s['alpha_ordinal']:.3f}.

Agreement by category (specific agreement): {spec}.
Most common disagreements ({rater_a} → {rater_b}): {top}.

Do the raters use the categories at the same rate? Stuart-Maxwell chi2({dof}) = {chi:.2f}, p {fmt_p(p)}.

| category | {rater_a} | {rater_b} |
| --- | --- | --- |
{marg_rows}

![{name}]({image})
"""
