"""Intent-classification evaluation on the frozen golden set: majority class vs TF-IDF + logistic regression vs Gemini.

    python scripts/evaluate_intents.py                        # all three systems (needs GEMINI_API_KEY for Gemini)
    python scripts/evaluate_intents.py --skip-gemini          # baselines only, no key needed
    python scripts/evaluate_intents.py --pause-seconds 7      # stay under a free-tier per-minute limit

Baselines are fitted on train_retrieval and selected on dev_calibration (weak cluster-derived labels); golden is only scored.
Gemini predictions are cached in reports/intent_eval/gemini_predictions.jsonl, keyed by case, model and prompt hash, so an
interrupted run resumes and a finished run re-scores without calling the API. Cases are sent in labeling order, so the 100
blind-human cases are covered first.

Outputs: reports/intent_eval/virgintrains_intent_evaluation.md and .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from common.logging_utils import configure_logging
from evaluation.golden_eval import GoldenLabelStore, GoldVocabulary
from evaluation.intent_eval import (
    ALL_REVIEWED,
    BLIND_HUMAN,
    MajorityBaseline,
    evaluate_systems,
    fit_tfidf_logreg,
    gemini_predictions,
    golden_eval_frame,
    load_prediction_cache,
    render_markdown,
    weak_labelled_split,
)
from evaluation.labeling_store import LabelStoreError
from evaluation.smoke_report import changed_files, fingerprint_files
from evaluation.splits import DEV, TRAIN
from taxonomy.registry import FALLBACK_INTENT, intent_names

ROOT = _bootstrap.REPO_ROOT


def build_gemini_classifier(registry: dict):
    from agent.classifier import IntentClassifier
    from agent.config import load_config
    from models.gemini import GeminiModel

    config = load_config(None)
    m = config.model
    model = GeminiModel(model_name=m.name, timeout_seconds=m.timeout_seconds, max_retries=m.max_retries, response_schema=m.response_schema)
    classifier = IntentClassifier(
        model, registry, examples_per_intent=config.classifier.examples_per_intent, temperature=m.classifier_temperature,
        max_output_tokens=m.max_output_tokens, use_schema=m.response_schema,
    )
    return classifier, model.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=ROOT / "data" / "golden")
    parser.add_argument("--processed-dir", type=Path, default=_bootstrap.DEFAULT_PROCESSED)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "virgintrains_intents.yaml")
    parser.add_argument("--cluster-labels", type=Path, default=ROOT / "configs" / "virgintrains_cluster_labels.yaml")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "intent_eval")
    parser.add_argument("--skip-gemini", action="store_true", help="Score only the baselines (and any cached Gemini predictions are ignored).")
    parser.add_argument("--pause-seconds", type=float, default=0.0, help="Wait between uncached Gemini calls.")
    parser.add_argument("--env-file", type=Path, default=_bootstrap.DEFAULT_ENV_FILE)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)
    _bootstrap.load_env_file(args.env_file)

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["taxonomy"]
    intents = intent_names(registry)
    splits = args.processed_dir / "splits"
    protected = [
        args.registry, args.cluster_labels, splits / "virgintrains_split_assignments.csv",
        args.golden_dir / "virgintrains_golden_v1.csv", args.golden_dir / "virgintrains_golden_v1_label_audit.jsonl",
    ]
    before = fingerprint_files(protected)
    try:
        store = GoldenLabelStore(
            args.golden_dir, GoldVocabulary.from_registry(registry), split_manifest_path=splits / "virgintrains_split_manifest.json",
            assignments_path=splits / "virgintrains_split_assignments.csv", registry_path=args.registry,
        )
    except LabelStoreError as exc:
        print(f"Refusing to evaluate: {exc}", file=sys.stderr)
        return 2
    golden = golden_eval_frame(store)
    scored = golden[golden[ALL_REVIEWED]]
    print(f"Golden: {len(golden)} cases; {int(golden[BLIND_HUMAN].sum())} blind-human, {len(scored)} reviewed in total.")

    train = weak_labelled_split(args.processed_dir, args.cluster_labels, TRAIN)
    dev = weak_labelled_split(args.processed_dir, args.cluster_labels, DEV)
    texts = scored["text"].tolist()
    majority = MajorityBaseline().fit(train["text"].tolist(), train["label"].tolist())
    tfidf = fit_tfidf_logreg(train, dev)
    predictions = {
        "majority_class": dict(zip(scored["case_id"], majority.predict(texts))),
        "tfidf_logreg": dict(zip(scored["case_id"], tfidf.model.predict(texts))),
    }
    print(f"Baselines fitted on {len(train):,} train cases; C={tfidf.c} chosen on {len(dev):,} dev cases.")

    report_dir = args.report_dir
    cache_path = report_dir / "gemini_predictions.jsonl"
    meta = {
        "train_cases": len(train), "dev_cases": len(dev), "majority_label": majority.label_, "tfidf_c": tfidf.c,
        "tfidf_dev_macro_f1": {str(c): round(v, 4) for c, v in tfidf.dev_macro_f1.items()},
        "provenance": ", ".join(f"{k}: {v}" for k, v in sorted(Counter(golden["provenance"]).items())),
        "gemini_model": None, "gemini_stopped": None, "gemini_unparseable": 0,
    }
    if not args.skip_gemini:
        from agent.classifier import ClassifierError
        from models.base import ModelConfigError, ModelError

        try:
            classifier, model_name = build_gemini_classifier(registry)
        except ModelConfigError as exc:
            print(f"SETUP FAILURE: {exc}\nRe-run with --skip-gemini to score the baselines only.", file=sys.stderr)
            return 2
        preds, stopped = gemini_predictions(
            classifier, scored, cache_path, model_name=model_name, fallback_intent=FALLBACK_INTENT,
            output_errors=(ClassifierError,), stop_errors=(ModelError,), pause_seconds=args.pause_seconds,
        )
        cached = [r for r in load_prediction_cache(cache_path).values() if r["model"] == model_name and r["case_id"] in preds]
        meta.update(gemini_model=model_name, gemini_stopped=stopped, gemini_unparseable=sum(1 for r in cached if r.get("error")))
        predictions["gemini"] = preds
        print(f"Gemini ({model_name}): {len(preds)} / {len(scored)} cases predicted." + (f" {stopped}" if stopped else ""))

    results = {"meta": meta, "subsets": evaluate_systems(golden, predictions, intents)}
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = report_dir / "virgintrains_intent_evaluation.md", report_dir / "virgintrains_intent_evaluation.json"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for subset, systems in results["subsets"].items():
        print(f"\n{subset}")
        for name, m in systems.items():
            if m["n"]:
                print(f"  {name:15s} n={m['n']:3d}  accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
    touched = changed_files(before, fingerprint_files(protected))
    if touched:
        print(f"\nINTEGRITY FAILURE: protected files changed during the run: {touched}", file=sys.stderr)
        return 1
    print(f"\nReport: {md_path}\nMetrics: {json_path}")
    return 1 if meta["gemini_stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
