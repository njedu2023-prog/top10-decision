#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from top10decision.ingest import load_latest_pred
from top10decision.decision_p0 import decision_p0
from top10decision.writers import write_latest_signal


def main():
    df = load_latest_pred()
    sig = decision_p0(df)
    out = write_latest_signal(sig)
    print(f"[run_v2] wrote {out}")


if __name__ == "__main__":
    main()
