"""Compare the candidate taxonomy with the human calibration labels.

Inputs:   data/processed/taxonomy_calibration.csv (human_* columns filled in by a person)
Outputs:  reports/taxonomy_calibration_report.md
          data/processed/taxonomy_calibration_report.json
          data/processed/taxonomy_calibration_confusion_matrix.csv
          data/processed/taxonomy_calibration_confusion_pairs.csv

Exits with status 2 (and writes nothing) if no case has a complete set of human labels. This script never
invents labels. The report is marked PRELIMINARY until at least 95% of the rows are fully labelled.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from common.logging_utils import configure_logging
from evaluation.taxonomy_comparison import Thresholds, compare, load_labelled
from taxonomy.registry import intent_names

logger = logging.getLogger("compare_taxonomy")


def pct(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.0f}%"


def md_table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render(result: dict, load, th: Thresholds, fallback: str) -> str:
    complete = load.n_complete / load.n_rows if load.n_rows else 0.0
    prelim = complete < th.complete_fraction
    L = ["# Taxonomy calibration: candidate intents vs human labels", ""]
    L.append(
        f"**Status: {'PRELIMINARY' if prelim else 'COMPLETE'}.** {load.n_complete} of {load.n_rows} calibration cases are fully labelled ({pct(complete)}). "
        "Cases come from the golden-pool reserve; no golden case is involved. The sample deliberately over-represents confusable intents, so only rates "
        "conditional on the candidate intent are meaningful; small cells are noisy."
    )
    if load.problems:
        L += ["", "## Label problems (fix before relying on this report)", "", *[f"- {p}" for p in load.problems[:40]]]
        if len(load.problems) > 40:
            L.append(f"- ... and {len(load.problems) - 40} more")
    o = result["overall"]
    L += [
        "",
        "## Overall",
        "",
        f"- Candidate = human intent: {pct(o['agreement'])} unweighted, {pct(o['agreement_weighted_to_reserve_eligible'])} re-weighted within intents; Cohen's kappa {o['cohens_kappa']:.2f}.",
        f"- {o['note']}",
        "",
        "## Intent purity (per candidate intent)",
        "",
        "`precision` = share of the candidate intent's cases that humans gave the same intent; `purity` = share held by its single most common human intent.",
        "",
        md_table(
            ["candidate intent", "n", "precision (95% CI)", "purity", "most common human intent", "human mix"],
            [
                [p["candidate_intent"] + (" ⚠ low support" if p["low_support"] else ""), p["n"], f"{pct(p['precision'])} ({pct(p['precision_ci95'][0])}-{pct(p['precision_ci95'][1])})",
                 pct(p["purity"]), p["modal_human_intent"], ", ".join(f"{k} {v}" for k, v in p["human_mix"].items())]
                for p in result["purity"]
            ],
        ),
        "",
        "## Coverage (per human intent)",
        "",
        "`recall` = share of the human intent's cases that the candidate taxonomy also gave that intent.",
        "",
        md_table(
            ["human intent", "n", "recall", "candidate mix"],
            [[r["human_intent"] + (" ⚠ low support" if r["low_support"] else ""), r["n"], pct(r["recall"]), ", ".join(f"{k} {v}" for k, v in r["candidate_mix"].items())] for r in result["recall"]],
        ),
        "",
        "## Confusion matrix (rows: candidate intent, columns: human intent)",
        "",
    ]
    m = result["confusion_matrix"]
    L.append(md_table(["candidate \\ human"] + list(m.columns) + ["total"], [[idx] + [int(v) for v in row] + [int(row.sum())] for idx, row in m.iterrows()]))
    L += ["", "## Systematic disagreements", "", f"Pairs of (candidate, human) with at least {th.min_pair} cases:", ""]
    if not result["systematic_disagreements"]:
        L.append("None reached the threshold.")
    for d in result["systematic_disagreements"]:
        L += [f"### `{d['candidate_intent']}` labelled `{d['human_intent']}` by humans ({d['n']} cases)", ""]
        L += [f"- `{e['case_id']}`: {e['message']}" + (f"  _human note: {e['human_notes']}_" if e["human_notes"] else "") for e in d["examples"]]
        L.append("")
    L += ["## Intents that should merge", "", f"Evidence: mutual confusion rate >= {pct(th.merge_rate)} with disagreements in both directions.", ""]
    if not result["merge_candidates"]:
        L.append("No pair reached the threshold.")
    for mg in result["merge_candidates"]:
        L.append(
            f"- `{mg['intents'][0]}` + `{mg['intents'][1]}`: {mg['a_labelled_as_b']} and {mg['b_labelled_as_a']} cases labelled as the other (rate {pct(mg['mutual_confusion_rate'])}); "
            f"human resolution-type overlap {pct(mg['human_resolution_type_overlap'])}; mentioned together in notes {mg['human_also_mentions']}x. {mg['reading']}"
        )
    L += ["", "## Intents that should split", "", f"Evidence: purity < {pct(th.split_purity)} and the second human intent holds >= {pct(th.split_share)}.", ""]
    if not result["split_candidates"]:
        L.append("No intent met the criteria.")
    for s in result["split_candidates"]:
        L.append(f"- `{s['candidate_intent']}` (n={s['n']}, purity {pct(s['purity'])}): human labels {s['human_mix_shares']}. {s['reading']}")
    L += ["", "## New intents proposed by the labeller", ""]
    if not result["proposed_new_intents"]:
        L.append("None.")
    for p in result["proposed_new_intents"]:
        L.append(f"- `{p['name']}` ({p['n']} cases, from {p['from_candidates']})")
        L += [f"  - `{e['case_id']}`: {e['message']}" + (f"  _{e['human_notes']}_" if e["human_notes"] else "") for e in p["examples"]]
    fq = result["fallback_quality"]
    L += [
        "",
        f"## Fallback quality (`{fallback}`)",
        "",
        f"- Candidate fallback cases: {fq['candidate_fallback_n']}; human fallback cases: {fq['human_fallback_n']}.",
        f"- Precision {pct(fq['precision'])} (share of candidate-fallback cases humans also could not assign); recall {pct(fq['recall'])} (share of human-unclear cases the system sent to fallback).",
        f"- Candidate fallback cases that humans assigned a real intent: {fq['candidate_fallback_cases_humans_assigned_real_intent'] or 'none'}.",
        f"- Human-unclear cases the system missed (candidate intent given): {fq['human_fallback_cases_candidate_missed'] or 'none'}.",
        f"- {fq['reading']}",
    ]
    if "resolution_type" in result:
        r, rs = result["resolution_type"], result["resolved"]
        L += [
            "",
            "## Resolution labels vs human",
            "",
            f"- Resolution type agreement {pct(r['agreement'])}. Resolved flag agreement {pct(rs['agreement'])} on {rs['n_comparable']} comparable cases "
            f"(auto yes / human no: {rs['auto_yes_human_no']}; auto no / human yes: {rs['auto_no_human_yes']}; human unclear: {rs['human_unclear']}).",
            "",
            md_table(
                ["auto resolution type", "n", "agreement", "human mix"],
                [[t, v["n"], pct(v["agreement"]), ", ".join(f"{k} {c}" for k, c in v["human_mix"].items())] for t, v in sorted(r["per_auto_type"].items(), key=lambda kv: -kv[1]["n"])],
            ),
        ]
    L += ["", "## Escalation signals by human intent", ""]
    L.append(md_table(["human intent", "signals"], [[k, ", ".join(f"{a} {b}" for a, b in v.items())] for k, v in sorted(result["escalation_signals_by_human_intent"].items())]))
    L += [
        "",
        "## Next step",
        "",
        "Read the merges, splits and fallback sections, then record your decisions in `configs/virgintrains_taxonomy_decisions.yaml` and run "
        "`python scripts/finalize_taxonomy.py`. The tool recommends; the reviewer decides.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=_bootstrap.DEFAULT_PROCESSED / "taxonomy_calibration.csv")
    parser.add_argument("--registry", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--out-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--reports-dir", type=Path, default=_bootstrap.REPO_ROOT / "reports")
    parser.add_argument("--min-support", type=int, default=Thresholds.min_support)
    parser.add_argument("--min-pair", type=int, default=Thresholds.min_pair)
    parser.add_argument("--merge-rate", type=float, default=Thresholds.merge_rate)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    taxonomy = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    names = intent_names(taxonomy)
    fallback = taxonomy["fallback"]["name"]
    raw = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    load = load_labelled(raw, names, fallback)
    th = Thresholds(min_support=args.min_support, min_pair=args.min_pair, merge_rate=args.merge_rate)
    if load.n_complete == 0:
        logger.error("No case in %s has a complete set of human labels (%d rows). Nothing to compare; labels are never auto-filled.", args.labels, load.n_rows)
        raise SystemExit(2)
    labelled = load.frame.copy()
    labelled["stratum_weight"] = pd.to_numeric(labelled["stratum_weight"], errors="coerce").fillna(1.0)
    result = compare(labelled, names, fallback, th)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "taxonomy_calibration_report.md").write_text(render(result, load, th, fallback), encoding="utf-8")
    result["confusion_matrix"].to_csv(args.out_dir / "taxonomy_calibration_confusion_matrix.csv")
    result["confusion_pairs"].to_csv(args.out_dir / "taxonomy_calibration_confusion_pairs.csv", index=False)
    serial = {k: v for k, v in result.items() if k not in ("confusion_matrix", "confusion_pairs")}
    if "resolution_type" in serial:
        serial["resolution_type"] = {k: v for k, v in serial["resolution_type"].items() if k != "confusion"}
    serial["label_problems"] = load.problems
    serial["n_rows"], serial["n_complete"] = load.n_rows, load.n_complete
    (args.out_dir / "taxonomy_calibration_report.json").write_text(json.dumps(serial, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote taxonomy_calibration_report.{md,json} (%d/%d labelled, %d problems)", load.n_complete, load.n_rows, len(load.problems))


if __name__ == "__main__":
    main()
