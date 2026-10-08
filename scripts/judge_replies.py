"""LLM-as-judge over the agent's draft replies, and judge-vs-human agreement.

    python scripts/judge_replies.py                      # judge with Groq (GROQ_MODEL or gpt-oss-120b; needs GROQ_API_KEY)
    python scripts/judge_replies.py --model openai/gpt-oss-20b

Reads reports/agent_eval/judge_inputs.jsonl (written by scripts/evaluate_agent.py). Every judge call is cached in
reports/agent_eval/judge_cache.jsonl, so reruns are free and an interrupted run (rate limit) resumes.

Human ratings: the first run writes reports/agent_eval/human_ratings.csv with up to 40 representative replies and blank
human_* columns (1-5, same rubric as printed in the report; judge scores are not shown). Fill it in and rerun; agreement is
computed on the fully rated rows. Once any rating is present the sheet is never overwritten.

Outputs in reports/agent_eval/: judge_evaluation.md and .json, judge_scores.jsonl, human_ratings.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from common.logging_utils import configure_logging
from evaluation.agent_eval import CachingModel
from evaluation.llm_judge import (
    HUMAN_COLUMNS,
    JudgeError,
    agreement,
    human_rating_sample,
    judge_item,
    judge_summary,
    load_human_ratings,
    load_items,
    render_markdown,
    write_human_template,
)
from evaluation.smoke_report import changed_files, fingerprint_files
from models.base import ModelConfigError, ModelRuntimeError

ROOT = _bootstrap.REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "agent_eval")
    parser.add_argument("--model", default=None, help="Groq model id (default: GROQ_MODEL or openai/gpt-oss-120b).")
    parser.add_argument("--max-human", type=int, default=40, help="Replies in the human rating sheet (default 40).")
    parser.add_argument("--env-file", type=Path, default=_bootstrap.DEFAULT_ENV_FILE)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)
    _bootstrap.load_env_file(args.env_file)

    report_dir = args.report_dir
    inputs_path = report_dir / "judge_inputs.jsonl"
    human_path = report_dir / "human_ratings.csv"
    try:
        items = load_items(inputs_path)
    except JudgeError as exc:
        print(f"Refusing to judge: {exc}", file=sys.stderr)
        return 2
    print(f"{len(items)} draft replies in {inputs_path.name}.")
    before = fingerprint_files([inputs_path])

    from models.groq import GroqModel

    try:
        model = CachingModel(GroqModel(model_name=args.model), report_dir / "judge_cache.jsonl")
    except ModelConfigError as exc:
        print(f"SETUP FAILURE: {exc}", file=sys.stderr)
        return 2

    scores, stopped = [], None
    for i, item in enumerate(items, 1):
        try:
            s = judge_item(model, item)
        except ModelRuntimeError as exc:
            stopped = f"stopped at reply {i}/{len(items)} ({item['case_id']}): {str(exc)[:300]}"
            break
        except ModelConfigError as exc:
            print(f"SETUP FAILURE: {exc}", file=sys.stderr)
            return 2
        scores.append(s)
        dims = " ".join(f"{k[:5]}={s[k]}" for k in ("correctness", "groundedness", "actionability", "brand_alignment"))
        print(f"[{i:02d}/{len(items)}] {item['case_id']:14s} {dims}" + (f"  ERROR {s['error']}" if s["error"] else ""), flush=True)

    try:
        sample = human_rating_sample(items, max_n=args.max_human)
        written = write_human_template(sample, human_path)
        print(f"Human rating sheet (blind, {written} replies): {human_path}")
        human = None
    except JudgeError:
        try:
            human = load_human_ratings(human_path)
        except JudgeError as exc:
            print(f"Cannot use human ratings: {exc}", file=sys.stderr)
            return 2
        print(f"Using {len(human)} fully rated replies from {human_path.name}.")
    agree = agreement(scores, human) if human is not None else None

    summary = judge_summary(items, scores)
    meta = {"provider": "groq", "model": model.name, "cache_hits": model.hits, "cache_misses": model.misses, "stopped": stopped, "human_csv": human_path.name}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "judge_evaluation.md").write_text(render_markdown(meta, summary, agree), encoding="utf-8")
    (report_dir / "judge_evaluation.json").write_text(json.dumps({"meta": meta, "judge": summary, "agreement": agree, "human_columns": HUMAN_COLUMNS}, indent=2), encoding="utf-8")
    with open(report_dir / "judge_scores.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in scores)

    print(f"\nJudged {len(scores)}/{len(items)}. Calls: {model.hits} cached, {model.misses} new." + (f"\n{stopped}" if stopped else ""))
    if agree and agree["n"]:
        for d in ("correctness", "groundedness", "actionability", "brand_alignment"):
            a = agree[d]
            print(f"  {d:16s} exact={a['exact_agreement']:.2f} kappa={a['weighted_kappa']} spearman={a['spearman']}")
    touched = changed_files(before, fingerprint_files([inputs_path]))
    if touched:
        print(f"\nINTEGRITY FAILURE: judge inputs changed during the run: {touched}", file=sys.stderr)
        return 1
    print(f"\nReport: {report_dir / 'judge_evaluation.md'}")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
