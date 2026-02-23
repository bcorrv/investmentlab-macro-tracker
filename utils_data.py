from __future__ import annotations

from io import StringIO
import json
import pandas as pd
import requests

OWNER = "bcorrv"
DATA_REPO = "investmentlab-data"
CODE_REPO = "investmentlab-macro-tracker"
BRANCH = "main"

DATA_URL = f"https://raw.githubusercontent.com/{OWNER}/{DATA_REPO}/{BRANCH}/daily_snapshot.csv"
RUN_SUMMARY_URL = f"https://raw.githubusercontent.com/{OWNER}/{CODE_REPO}/{BRANCH}/data/run_summary.json"


def _get_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def load_snapshot() -> pd.DataFrame:
    txt = _get_text(DATA_URL)
    df = pd.read_csv(StringIO(txt))
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")
    return df


def load_run_summary() -> dict:
    try:
        txt = _get_text(RUN_SUMMARY_URL)
        return json.loads(txt)
    except Exception:
        return {}