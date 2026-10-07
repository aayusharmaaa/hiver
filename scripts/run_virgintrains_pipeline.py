"""Run the whole VirginTrains data pipeline in dependency order.

    build -> discover -> splits -> taxonomy -> report

Each step is an ordinary script, so any of them can be re-run alone. Use --from/--to to run a slice,
e.g. `--from taxonomy` after hand-editing configs/virgintrains_cluster_labels.yaml.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from common.logging_utils import configure_logging

logger = logging.getLogger("run_virgintrains_pipeline")
SCRIPTS = Path(__file__).resolve().parent
STEPS = {
    "build": "build_virgintrains.py",
    "discover": "discover_intents.py",
    "splits": "prepare_splits.py",
    "taxonomy": "generate_taxonomy.py",
    "report": "virgintrains_report.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=_bootstrap.DEFAULT_RAW, help="Path to twcs.csv (used by the build step).")
    parser.add_argument("--from", dest="first", choices=STEPS, default="build")
    parser.add_argument("--to", dest="last", choices=STEPS, default="report")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    names = list(STEPS)
    selected = names[names.index(args.first) : names.index(args.last) + 1]
    for name in selected:
        cmd = [sys.executable, str(SCRIPTS / STEPS[name])]
        if name == "build":
            cmd += ["--input", str(args.input)]
        logger.info("=== %s: %s", name, " ".join(cmd[1:]))
        started = time.time()
        subprocess.run(cmd, check=True)
        logger.info("=== %s finished in %.0fs", name, time.time() - started)


if __name__ == "__main__":
    main()
