"""Loading rating data in the two shapes it usually arrives in."""

import numpy as np


def table_to_pairs(table):
    """Expand a two-rater contingency table (DataFrame: rows rater 1, cols rater 2) into label pairs."""
    r1, r2 = [], []
    for a in table.index:
        for b in table.columns:
            n = int(table.loc[a, b])
            r1 += [a] * n
            r2 += [b] * n
    return np.array(r1), np.array(r2)


def long_to_matrix(df, item="item", rater="rater", label="label", order=None):
    """Long format (one row per rating) -> raters x items matrix of numeric codes, NaN when missing.

    order: list of labels in their natural order (needed for ordinal/interval metrics).
    """
    labels = order or sorted(df[label].unique(), key=str)
    code = {v: i for i, v in enumerate(labels)}
    wide = df.assign(_c=df[label].map(code)).pivot_table(index=rater, columns=item, values="_c", aggfunc="first")
    return wide.to_numpy(dtype=float), labels


def matrix_to_counts(matrix, n_categories):
    """Raters x items codes -> items x categories counts (for Fleiss' kappa)."""
    m = np.asarray(matrix, float)
    counts = np.zeros((m.shape[1], n_categories))
    for j in range(m.shape[1]):
        for v in m[:, j][~np.isnan(m[:, j])]:
            counts[j, int(v)] += 1
    return counts
