"""Build the candidate cluster report and draft taxonomy YAML.

Outputs:
    data/processed/virgintrains_intent_clusters.json
    reports/virgintrains_intent_clusters.md
    configs/virgintrains_intents.yaml

Run after discover_intents.py and prepare_splits.py.
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
from taxonomy.registry import STATUS_CALIBRATED, registry_status
from taxonomy.taxonomy_builder import build_cluster_report, build_cluster_table, build_taxonomy, match_labels

logger = logging.getLogger("generate_taxonomy")


def render_cluster_markdown(report: list[dict], preview: dict) -> str:
    lines = [
        "# VirginTrains candidate intent clusters",
        "",
        f"Embedding model `{preview['embedding_model']}`, KMeans k={preview['k']}, fit on {preview['n_fit']:,} train+dev openers "
        f"(entities masked). TF-IDF/LSA vs embedding agreement: NMI {preview['tfidf_agreement']['nmi']:.2f}, ARI {preview['tfidf_agreement']['ari']:.2f}.",
        "These are **candidate** clusters; names and descriptions are hand-written after reading examples. Examples come from the train split.",
        "",
        "| id | proposed intent | cases | % resolved | % DM | final intent |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in sorted(report, key=lambda x: -x["n_cases"]):
        lines.append(
            f"| {r['cluster_id']} | {r['proposed_intent_name']} | {r['n_cases']:,} | {100 * r['pct_resolved']:.1f}% | "
            f"{100 * r['pct_dm_redirect']:.1f}% | {r['recommended_final_intent']} |"
        )
    for r in sorted(report, key=lambda x: -x["n_cases"]):
        neighbors = ", ".join(
            "{} {} ({:.0f}%)".format(n["cluster_id"], n["name"], 100 * n["share_of_low_margin_cases"])
            for n in r["confusable_neighbors"]
        ) or "n/a"
        lines += [
            "",
            f"## Cluster {r['cluster_id']}: {r['proposed_intent_name']}",
            "",
            r["description"],
            "",
            f"- Cases: {r['n_cases']:,} ({100 * r['share_of_clustered']:.1f}% of clustered); resolved {100 * r['pct_resolved']:.1f}%; DM redirect {100 * r['pct_dm_redirect']:.1f}%",
            f"- Resolution types: {', '.join(f'{k} {100 * v:.0f}%' for k, v in r['resolution_type_mix'].items())}",
            f"- Confusable with: {neighbors}",
            f"- Auto-handle hint: `{r['auto_handle_hint']}` (NEEDS_REVIEW)",
            "",
            "Representative examples:",
            *[f"  - `{e['case_id']}` {e['text']}" for e in r["representative_examples"]],
            "",
            "Ambiguous examples:",
            *[f"  - `{e['case_id']}` {e['text']}  _(also close to {e['also_close_to']})_" for e in r["ambiguous_examples"]],
            "",
            "Representative resolution patterns (first agent reply):",
            *([f"  - {p['count']}x \"{p['example']}\"" for p in r["representative_resolution_patterns"]] or ["  - none recurring (>=3)"]),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--labels", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--out-yaml", type=Path, default=_bootstrap.REPO_ROOT / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--reports-dir", type=Path, default=_bootstrap.REPO_ROOT / "reports")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    cases = pd.read_parquet(args.processed_dir / "virgintrains_cases.parquet", columns=[
        "case_id", "opening_message", "resolved", "dm_redirect", "resolution_type", "resolution_signals", "agent_messages", "turn_count"])
    clusters = pd.read_parquet(args.processed_dir / "virgintrains_case_clusters.parquet")
    assignments = pd.read_csv(args.processed_dir / "splits" / "virgintrains_split_assignments.csv")
    preview = json.loads((args.processed_dir / "virgintrains_cluster_preview.json").read_text(encoding="utf-8"))
    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))

    terms = {int(c): v["top_terms"] for c, v in preview["clusters"].items()}
    specs = match_labels(terms, labels["clusters"])
    table = build_cluster_table(cases, clusters, assignments)
    report = build_cluster_report(table, specs)
    taxonomy = build_taxonomy(
        table, specs, report, labels.get("special", []),
        final_descriptions=labels.get("final_intents"),
        criteria=labels.get("intent_criteria"),
        confusable_notes=labels.get("confusable_notes"),
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.processed_dir / "virgintrains_intent_clusters.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.reports_dir / "virgintrains_intent_clusters.md").write_text(render_cluster_markdown(report, preview), encoding="utf-8")
    out = args.out_yaml
    if registry_status(out) == STATUS_CALIBRATED:
        out = out.with_name(out.stem + ".candidate.yaml")
        logger.warning("%s is HUMAN_CALIBRATED and is never overwritten here; writing the regenerated candidate to %s", args.out_yaml, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "# DRAFT candidate taxonomy for VirginTrains. NOT ground truth. Generated by scripts/generate_taxonomy.py.\n"
    out.write_text(header + yaml.safe_dump(taxonomy, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    logger.info("Wrote %d clusters and %d draft intents + fallback to %s", len(report), len(taxonomy["taxonomy"]["intents"]), out)


if __name__ == "__main__":
    main()
