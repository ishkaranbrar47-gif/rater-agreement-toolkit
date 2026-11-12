"""Inter-rater agreement reports.

Examples:
    # the two clinical demos shipped in data/
    python agree.py demo --out reports

    # your own annotation data: one row per rating with item, rater, label columns
    python agree.py annotations ratings.csv --order bad ok good --level ordinal
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agreement.io import long_to_matrix, matrix_to_counts
from agreement.metrics import fleiss_kappa, interpret_kappa, krippendorff_alpha
from agreement.report import heatmap, markdown_section, summarize


def load_ms_patients(path):
    """vcd::MSPatients flattened: rows = New Orleans neurologist, columns = Winnipeg neurologist x patient site."""
    raw = pd.read_csv(path, index_col=0)
    sites = {}
    for site in ("Winnipeg", "New Orleans"):
        cols = [c for c in raw.columns if c.endswith("." + site)]
        t = raw[cols]
        t.columns = [c.split(".")[0] for c in cols]
        sites[site] = t
    return sites


def run_demo(data_dir, out_dir):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sections, results = [], {}

    mammo = pd.read_csv(data / "mammograms.csv", index_col=0)
    s = summarize(mammo, "Radiologist 1", "Radiologist 2")
    heatmap(mammo, "Mammogram ratings (n = 110)", "Radiologist 1", "Radiologist 2", out / "mammograms.png")
    sections.append(markdown_section("Mammograms: two radiologists", s, "Radiologist 1", "Radiologist 2", "mammograms.png"))
    results["mammograms"] = s

    for site, t in load_ms_patients(data / "ms_patients.csv").items():
        s = summarize(t, "New Orleans neurologist", "Winnipeg neurologist")
        fname = f"ms_{site.lower().replace(' ', '_')}.png"
        heatmap(t, f"MS diagnosis, {site} patients (n = {s['n']})", "New Orleans neurologist", "Winnipeg neurologist",
                out / fname)
        sections.append(markdown_section(f"Multiple sclerosis: {site} patients", s, "New Orleans neurologist",
                                         "Winnipeg neurologist", fname))
        results[f"ms_{site}"] = s

    (out / "report.md").write_text("# Inter-Rater Agreement Report\n\n" + "\n".join(sections))
    return results


def run_annotations(path, order=None, level="nominal"):
    """Agreement for any number of annotators, plus the items they disagree on most."""
    df = pd.read_csv(path)
    matrix, labels = long_to_matrix(df, order=order)
    alpha = krippendorff_alpha(matrix, level=level)
    out = {"items": matrix.shape[1], "raters": matrix.shape[0], "alpha": alpha}
    complete = ~np.isnan(matrix).any(axis=0)
    if complete.sum() > 0:
        out["fleiss"] = fleiss_kappa(matrix_to_counts(matrix[:, complete], len(labels)))
    items = df["item"].drop_duplicates().sort_values().tolist()
    disagreement = []
    for j, item in enumerate(items):
        col = matrix[:, j][~np.isnan(matrix[:, j])]
        if len(col) > 1:
            share = np.bincount(col.astype(int)).max() / len(col)
            disagreement.append((item, round(1 - share, 2)))
    out["flagged"] = sorted([d for d in disagreement if d[1] > 0], key=lambda x: -x[1])[:10]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Inter-rater agreement for clinical ratings and annotation QA.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="analyse the two clinical datasets in data/")
    d.add_argument("--data", default="data")
    d.add_argument("--out", default="reports")
    a = sub.add_parser("annotations", help="analyse a long-format CSV with item, rater, label columns")
    a.add_argument("csv")
    a.add_argument("--order", nargs="*", help="labels in their natural order")
    a.add_argument("--level", default="nominal", choices=["nominal", "ordinal", "interval"])
    args = p.parse_args(argv)

    if args.cmd == "demo":
        res = run_demo(args.data, args.out)
        for name, s in res.items():
            k = s["kappas"]["unweighted"][0]
            q = s["kappas"]["quadratic"][0]
            print(f"{name:<16} n={s['n']:<4} kappa={k:.3f} ({interpret_kappa(k)})  weighted={q:.3f} ({interpret_kappa(q)})")
        print(f"Report written to {Path(args.out) / 'report.md'}")
    else:
        res = run_annotations(args.csv, args.order, args.level)
        print(f"{res['raters']} raters, {res['items']} items")
        print(f"Krippendorff's alpha ({args.level}): {res['alpha']:.3f} ({interpret_kappa(res['alpha'])})")
        if "fleiss" in res:
            print(f"Fleiss' kappa (complete items): {res['fleiss']:.3f}")
        for item, d in res["flagged"]:
            print(f"  review item {item}: {d:.0%} of ratings disagree with the majority")


if __name__ == "__main__":
    main()
