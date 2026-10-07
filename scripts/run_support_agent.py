"""Run the support agent on one message.

    python scripts/run_support_agent.py --message "Is there wifi on the 09:00 to Glasgow?"
    python scripts/run_support_agent.py --message "yes please" --context "Customer: Can I take a bike on the train?"

Needs a Gemini API key in the GEMINI_API_KEY environment variable (GEMINI_MODEL optionally overrides the model name).
Intents come from the CANDIDATE taxonomy (not human validated); evidence comes from the train-split resolution memory.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import _bootstrap  # noqa: F401

from agent.support_agent import SupportAgent
from common.logging_utils import configure_logging
from models.base import ModelConfigError, ModelError

WIDTH = 100


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(" ".join((text or "").split()), width=WIDTH, initial_indent=indent, subsequent_indent=indent) or f"{indent}-"


def render(result, evidence_chars: int = 160) -> str:
    c, d = result.classification, result.decision
    out = [
        f"INTENT:      {c.intent}" + (f"   (alternative: {c.alternative_intent})" if c.alternative_intent else "") + ("   [multi-intent]" if c.multi_intent else ""),
        f"CONFIDENCE:  {c.confidence:.2f}   (model self-report; candidate taxonomy, not human validated)",
    ]
    if c.rationale:
        out.append(f"RATIONALE:   {c.rationale}")
    out.append(f"EVIDENCE:    {len(result.evidence)} retrieved case(s)")
    for i, e in enumerate(result.evidence, 1):
        sim = f"{e.semantic_similarity:.2f}" if e.semantic_similarity is not None else "n/a"
        out.append(f"  {i}. {e.case_id}  score={e.score:.3f}  similarity={sim}  type={e.resolution_type}  intent={e.intent}  tweets={','.join(e.source_tweet_ids) or '-'}")
        out.append(_wrap("problem:  " + e.customer_problem[:evidence_chars], "       "))
        out.append(_wrap("response: " + e.historical_response[:evidence_chars], "       "))
    out.append(f"DECISION:    {d.action}   (confidence {d.confidence:.2f})")
    out.append("REASONS:")
    out += [_wrap("- " + r, "  ") for r in d.reasons]
    out.append("REPLY:")
    out.append(_wrap(result.reply, "  ") if result.reply else "  (none - escalated to a human)")
    if result.internal_draft_reply:
        out.append("INTERNAL DRAFT (failed grounding; NOT to be sent):")
        out.append(_wrap(result.internal_draft_reply, "  "))
    if result.grounding is None:
        out.append("GROUNDING:   not run")
    else:
        g = result.grounding
        out.append(f"GROUNDING:   {'grounded' if g.grounded else 'NOT grounded'}   (verifier confidence {g.confidence:.2f})")
        out += [_wrap("- unsupported: " + u, "  ") for u in g.unsupported_claims]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--message", required=True, help="The customer message.")
    parser.add_argument("--context", default=None, help="Optional earlier conversation text.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (default: configs/support_agent.yaml).")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the full AgentResult as JSON instead.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(args.log_level)
    try:
        agent = SupportAgent.from_config(args.config, processed_dir=args.processed_dir)
        result = agent.handle(args.message, args.context)
    except ModelConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (ModelError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    logging.captureWarnings(True)
    sys.exit(main())
