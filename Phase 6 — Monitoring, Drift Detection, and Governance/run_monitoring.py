"""Operational monitoring CLI.

Usage:
  python run_monitoring.py validate <batch.csv> [--model risk_model|claim_model]
  python run_monitoring.py drift                 # regenerate the drift report
  python run_monitoring.py audit <audit.log>     # summarize an audit trail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from monitoring.audit_log import summarize_log
from monitoring.data_validation import validate_batch


def cmd_validate(args) -> int:
    df = pd.read_csv(args.path)
    report = validate_batch(df, model_key=args.model)
    print(report.summary())
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def cmd_drift(_args) -> int:
    import generate_drift_report
    generate_drift_report.main()
    return 0


def cmd_audit(args) -> int:
    summary = summarize_log(Path(args.path))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 monitoring CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate an incoming batch CSV")
    p_val.add_argument("path")
    p_val.add_argument("--model", default="claim_model",
                       choices=["risk_model", "claim_model"])
    p_val.add_argument("--json", action="store_true")
    p_val.set_defaults(func=cmd_validate)

    p_drift = sub.add_parser("drift", help="regenerate the drift detection report")
    p_drift.set_defaults(func=cmd_drift)

    p_audit = sub.add_parser("audit", help="summarize a prediction audit log")
    p_audit.add_argument("path")
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
