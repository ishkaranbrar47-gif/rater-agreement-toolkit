"""Agreement metrics implemented from their definitions.

Each function is checked in the tests against an independent library
(scikit-learn, statsmodels, krippendorff) so the maths can be trusted.
"""

import numpy as np


def _categories(*label_lists):
    cats = sorted(set().union(*[set(x) for x in label_lists]), key=str)
    return {c: i for i, c in enumerate(cats)}


def confusion(r1, r2, categories=None):
    """Square table: rows = rater 1's label, columns = rater 2's label."""
    if len(r1) != len(r2):
        raise ValueError("Both raters must label the same items.")
    idx = {c: i for i, c in enumerate(categories)} if categories is not None else _categories(r1, r2)
    table = np.zeros((len(idx), len(idx)))
    for a, b in zip(r1, r2, strict=True):
        table[idx[a], idx[b]] += 1
    return table


def percent_agreement(r1, r2):
    r1, r2 = np.asarray(r1), np.asarray(r2)
    return float((r1 == r2).mean())


def _weight_matrix(k, weights):
    i, j = np.indices((k, k))
    if weights is None:
        return (i != j).astype(float)
    if weights == "linear":
        return np.abs(i - j) / (k - 1)
    if weights == "quadratic":
        return ((i - j) / (k - 1)) ** 2
    raise ValueError("weights must be None, 'linear' or 'quadratic'")


def cohens_kappa(r1=None, r2=None, table=None, weights=None):
    """Cohen's kappa for two raters, from labels or an existing confusion table.

    weights=None treats every disagreement the same; 'linear' or 'quadratic'
    give partial credit for near-misses on ordered categories.
    """
    if table is None:
        table = confusion(r1, r2)
    t = np.asarray(table, float)
    n = t.sum()
    w = _weight_matrix(t.shape[0], weights)
    observed = (w * t).sum() / n
    expected = (w * np.outer(t.sum(axis=1), t.sum(axis=0))).sum() / n**2
    return float(1 - observed / expected)


def fleiss_kappa(counts):
    """Fleiss' kappa for any number of raters.

    counts: items x categories array; counts[i, j] = how many raters put item i in category j.
    Every item must have the same number of ratings.
    """
    c = np.asarray(counts, float)
    n_raters = c.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Every item needs the same number of ratings for Fleiss' kappa.")
    m = n_raters[0]
    p_j = c.sum(axis=0) / c.sum()
    p_i = ((c**2).sum(axis=1) - m) / (m * (m - 1))
    p_bar, p_e = p_i.mean(), (p_j**2).sum()
    return float((p_bar - p_e) / (1 - p_e))


def krippendorff_alpha(reliability_data, level="nominal"):
    """Krippendorff's alpha. reliability_data: raters x items, np.nan for missing.

    Handles any number of raters and missing ratings, which is common in real
    annotation work where not every annotator sees every item.
    """
    data = np.asarray(reliability_data, float)
    values = np.unique(data[~np.isnan(data)])
    v_idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))  # coincidence matrix
    for unit in data.T:
        u = unit[~np.isnan(unit)]
        m_u = len(u)
        if m_u < 2:
            continue
        for a in range(m_u):
            for b in range(m_u):
                if a != b:
                    o[v_idx[u[a]], v_idx[u[b]]] += 1 / (m_u - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if level == "nominal":
        delta = 1 - np.eye(k)
    elif level == "interval":
        delta = (values[:, None] - values[None, :]) ** 2
    elif level == "ordinal":
        cum = np.cumsum(n_c)
        delta = np.zeros((k, k))
        for g in range(k):
            for h in range(k):
                lo, hi = min(g, h), max(g, h)
                s = cum[hi] - (cum[lo - 1] if lo > 0 else 0)
                delta[g, h] = (s - (n_c[g] + n_c[h]) / 2) ** 2
    else:
        raise ValueError("level must be 'nominal', 'ordinal' or 'interval'")
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1 - d_o / d_e)


def bootstrap_ci(r1, r2, stat=cohens_kappa, n_boot=2000, level=0.95, seed=0, **kwargs):
    """Percentile bootstrap confidence interval, resampling items."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    cats = list(_categories(r1, r2))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(r1), len(r1))
        t = confusion(r1[i], r2[i], categories=cats)
        vals.append(stat(table=t, **kwargs))
    lo, hi = np.nanpercentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def interpret_kappa(k):
    """Landis & Koch (1977) labels."""
    if k < 0:
        return "poor"
    for cut, label in [(0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if k <= cut:
            return label
    return "almost perfect"
