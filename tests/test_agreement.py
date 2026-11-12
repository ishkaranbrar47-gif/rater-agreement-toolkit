import krippendorff
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss

import agree
from agreement.io import long_to_matrix, matrix_to_counts, table_to_pairs
from agreement.metrics import (
    bootstrap_ci,
    cohens_kappa,
    confusion,
    fleiss_kappa,
    interpret_kappa,
    krippendorff_alpha,
    percent_agreement,
)

RNG = np.random.default_rng(42)


def noisy_raters(n_items=200, n_raters=2, k=4, noise=0.3):
    truth = RNG.integers(0, k, n_items)
    raters = []
    for _ in range(n_raters):
        flip = RNG.random(n_items) < noise
        raters.append(np.where(flip, RNG.integers(0, k, n_items), truth))
    return np.array(raters)


# --- each metric matches an independent library -------------------------------------


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_cohens_kappa_matches_sklearn(weights):
    a, b = noisy_raters()
    assert cohens_kappa(a, b, weights=weights) == pytest.approx(cohen_kappa_score(a, b, weights=weights))


def test_fleiss_matches_statsmodels():
    data = noisy_raters(n_raters=5)
    counts, _ = aggregate_raters(data.T)
    assert fleiss_kappa(counts) == pytest.approx(sm_fleiss(counts))


@pytest.mark.parametrize("level", ["nominal", "ordinal", "interval"])
def test_krippendorff_matches_reference_package(level):
    data = noisy_raters(n_raters=4).astype(float)
    data[RNG.random(data.shape) < 0.15] = np.nan  # missing ratings are normal in annotation work
    expected = krippendorff.alpha(reliability_data=data, level_of_measurement=level)
    assert krippendorff_alpha(data, level) == pytest.approx(expected)


# --- known values on the real clinical data --------------------------------------------


@pytest.fixture
def mammograms():
    return pd.read_csv("data/mammograms.csv", index_col=0)


def test_mammogram_kappas_match_textbook_values(mammograms):
    # Same values as vcd::Kappa(Mammograms) in R
    assert cohens_kappa(table=mammograms.values) == pytest.approx(0.3713, abs=1e-4)
    assert cohens_kappa(table=mammograms.values, weights="linear") == pytest.approx(0.5964, abs=1e-4)


def test_table_to_pairs_round_trip(mammograms):
    r1, r2 = table_to_pairs(mammograms)
    assert len(r1) == 110
    rebuilt = confusion(r1, r2, categories=list(mammograms.index))
    assert np.array_equal(rebuilt, mammograms.values)


def test_ms_patients_split_by_site():
    sites = agree.load_ms_patients("data/ms_patients.csv")
    assert int(sites["Winnipeg"].values.sum()) == 149
    assert int(sites["New Orleans"].values.sum()) == 69


# --- edge cases ------------------------------------------------------------------------


def test_perfect_agreement():
    a = [0, 1, 2, 3, 1, 2]
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert percent_agreement(a, a) == 1.0


def test_kappa_below_zero_when_worse_than_chance():
    assert cohens_kappa([0, 1, 0, 1], [1, 0, 1, 0]) < 0


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        confusion([1, 2], [1])


def test_fleiss_rejects_unequal_rater_counts():
    with pytest.raises(ValueError):
        fleiss_kappa([[2, 1], [1, 1]])


def test_bad_weights_rejected():
    with pytest.raises(ValueError):
        cohens_kappa([0, 1], [0, 1], weights="cubic")


def test_bootstrap_ci_brackets_estimate():
    a, b = noisy_raters()
    lo, hi = bootstrap_ci(a, b, n_boot=500)
    assert lo < cohens_kappa(a, b) < hi


@pytest.mark.parametrize(
    "k, label", [(-0.1, "poor"), (0.1, "slight"), (0.3, "fair"), (0.5, "moderate"), (0.7, "substantial"), (0.9, "almost perfect")]
)
def test_landis_koch_labels(k, label):
    assert interpret_kappa(k) == label


# --- long-format annotation data -----------------------------------------------------


def test_long_format_with_missing_ratings():
    df = pd.DataFrame(
        {
            "item": [1, 1, 1, 2, 2, 3, 3, 3],
            "rater": ["a", "b", "c", "a", "b", "a", "b", "c"],
            "label": ["good", "good", "bad", "ok", "ok", "bad", "bad", "bad"],
        }
    )
    m, labels = long_to_matrix(df, order=["bad", "ok", "good"])
    assert m.shape == (3, 3)
    assert np.isnan(m[2, 1])  # rater c skipped item 2
    counts = matrix_to_counts(m, len(labels))
    assert counts.sum() == 8


def test_cli_annotations_flags_disputed_items(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "item": [1, 1, 1, 2, 2, 2],
            "rater": ["a", "b", "c"] * 2,
            "label": ["good", "good", "bad", "ok", "ok", "ok"],
        }
    )
    path = tmp_path / "ratings.csv"
    df.to_csv(path, index=False)
    agree.main(["annotations", str(path), "--order", "bad", "ok", "good", "--level", "ordinal"])
    out = capsys.readouterr().out
    assert "3 raters, 2 items" in out
    assert "review item 1" in out
    assert "review item 2" not in out


def test_cli_demo_writes_report(tmp_path, capsys):
    agree.main(["demo", "--data", "data", "--out", str(tmp_path)])
    for name in ("report.md", "mammograms.png", "ms_winnipeg.png", "ms_new_orleans.png"):
        assert (tmp_path / name).exists(), name
    assert "mammograms" in capsys.readouterr().out
