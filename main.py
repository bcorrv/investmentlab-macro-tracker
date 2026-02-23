from __future__ import annotations

from pathlib import Path
import time
import subprocess
import json
from datetime import datetime, timezone

import requests
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_URL = "https://mindicador.cl/api/"

DATA_DIR = Path("data")
CHART_DIR = Path("charts")
LOG_DIR = Path("logs")

DATA_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

INDICATORS = {
    "dolar": {"value_col": "usd_clp", "source": "mindicador"},
    "uf": {"value_col": "uf", "source": "mindicador"},
    "tpm": {"value_col": "tpm", "source": "mindicador"},
    "dxy": {"value_col": "dxy", "source": "fred", "series_id": "DTWEXBGS"},
}

ALERTS = {
    # Nivel absoluto
    "dolar_level": {"enabled": True, "threshold": 950.0},
    # Shock diario (%)
    "dolar_shock": {"enabled": True, "pct_threshold": 1.0},
    # Cruce MA30
    "dolar_ma30_cross": {"enabled": True},
    # Cambio de tasa (TPM)
    "tpm_change": {"enabled": True},
}

SNAPSHOT_PATH = DATA_DIR / "daily_snapshot.csv"
ALERT_LOG_PATH = Path("alerts.log")


# =========================
# UTILS
# =========================
def notify_mac(title: str, message: str) -> None:
    # En GitHub Actions (Linux) 'osascript' no existe. No rompemos el pipeline por alertas locales.
    if subprocess.call(["uname"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        try:
            # Sólo intenta en macOS
            if subprocess.check_output(["uname"], text=True).strip() == "Darwin":
                script = f'display notification "{message}" with title "{title}"'
                subprocess.run(["osascript", "-e", script], check=False)
        except Exception:
            pass


def log_line(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_json_with_retry(url: str, retries: int = 3, sleep_sec: float = 1.5) -> dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"Falló fetch tras {retries} intentos: {url}. Último error: {last_err}")


def fetch_csv_with_retry(url: str, retries: int = 3, sleep_sec: float = 1.5) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"Falló fetch CSV tras {retries} intentos: {url}. Último error: {last_err}")


def fetch_fred_graph_series(series_id: str, value_name: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    csv_text = fetch_csv_with_retry(url)

    from io import StringIO

    df = pd.read_csv(StringIO(csv_text))

    # Normaliza nombres
    df.columns = [c.strip().lower() for c in df.columns]

    # FRED puede traer 'date' o 'observation_date'
    date_col = None
    for candidate in ("date", "observation_date"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError(f"FRED CSV no trae columna fecha. Columnas: {df.columns.tolist()}")

    # La columna de valores suele ser la otra (ej: dtwexbgs, dgs10, etc.)
    value_col = series_id.lower()
    if value_col not in df.columns:
        # fallback: toma la 2da columna
        value_col = [c for c in df.columns if c != date_col][0]

    df = df.rename(columns={date_col: "fecha", value_col: value_name})

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["fecha", value_name]).sort_values("fecha")
    return df


def get_git_sha_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_run_summary(dfs: dict[str, pd.DataFrame], statuses: dict[str, str], out_path: Path = DATA_DIR / "run_summary.json") -> None:
    summary = {
        "pipeline_run_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": get_git_sha_short(),
        "indicators": {},
    }

    for ind, df in dfs.items():
        last_date = None
        if df is not None and len(df) > 0 and "fecha" in df.columns:
            last_date = str(df["fecha"].max())

        summary["indicators"][ind] = {
            "status": statuses.get(ind, "unknown"),
            "last_date": last_date,
        }

    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================
# DATA PIPELINE
# =========================
def fetch_series(indicator: str, value_name: str) -> pd.DataFrame:
    url = BASE_URL + indicator
    data = fetch_json_with_retry(url)

    serie = data.get("serie", [])
    if not serie:
        raise ValueError(f"No se encontraron datos para {indicator}")

    df = pd.DataFrame(serie)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True).dt.date
    df = df.rename(columns={"valor": value_name})
    df = df[["fecha", value_name]].sort_values("fecha")
    return df


def load_existing(path: Path, value_name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None

    # 1) Intentar UTF-8
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        # 2) Fallback típico cuando Excel mete Latin-1/Windows-1252
        df = pd.read_csv(path, encoding="latin1")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["fecha", value_name]).sort_values("fecha")
    return df


def merge_history(df_old: pd.DataFrame | None, df_new: pd.DataFrame) -> pd.DataFrame:
    if df_old is None:
        return df_new
    df = pd.concat([df_old, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=["fecha"], keep="last").sort_values("fecha")
    return df


def process_indicator(indicator: str, cfg: dict, run_log: Path, statuses: dict[str, str]) -> pd.DataFrame:
    value_name = cfg["value_col"]
    hist_path = DATA_DIR / f"{indicator}_daily.csv"
    latest_path = DATA_DIR / f"{indicator}_latest.csv"

    df_old = load_existing(hist_path, value_name)

    try:
        if cfg.get("source") == "fred":
            df_new = fetch_fred_graph_series(cfg["series_id"], value_name)
        else:
            df_new = fetch_series(indicator, value_name)  # mindicador

        df = merge_history(df_old, df_new)

        df.to_csv(hist_path, index=False, encoding="utf-8")
        pd.DataFrame([df.iloc[-1]]).to_csv(latest_path, index=False, encoding="utf-8")

        statuses[indicator] = "ok"
        return df

    except Exception as e:
        # Si falla la fuente, no botamos el pipeline: usamos el histórico (si existe)
        if df_old is not None and len(df_old) > 0:
            msg = f"WARN {indicator}: fetch falló ({type(e).__name__}) usando histórico"
            log_line(run_log, msg)
            pd.DataFrame([df_old.iloc[-1]]).to_csv(latest_path, index=False, encoding="utf-8")
            statuses[indicator] = "warn"
            return df_old

        statuses[indicator] = "fail"
        raise


# =========================
# SIGNALS / ALERTS
# =========================
def check_level(df: pd.DataFrame, value_name: str, indicator: str, threshold: float) -> None:
    latest_value = float(df.iloc[-1][value_name])
    latest_date = df.iloc[-1]["fecha"]
    if latest_value > threshold:
        msg = f"{indicator.upper()} {latest_value:.2f} el {latest_date} (umbral {threshold})"
        log_line(ALERT_LOG_PATH, "ALERTA: " + msg)
        notify_mac("Investment Lab — Alerta", msg)


def check_daily_change(df: pd.DataFrame, value_name: str, indicator: str, pct_threshold: float) -> None:
    if len(df) < 2:
        return
    prev = float(df.iloc[-2][value_name])
    latest = float(df.iloc[-1][value_name])
    latest_date = df.iloc[-1]["fecha"]
    if prev == 0:
        return

    pct = (latest / prev - 1.0) * 100.0
    if abs(pct) >= pct_threshold:
        direction = "SUBE" if pct > 0 else "BAJA"
        msg = f"{indicator.upper()} {direction} {pct:.2f}% el {latest_date}"
        log_line(ALERT_LOG_PATH, "ALERTA: " + msg)
        notify_mac("Investment Lab — Shock diario", msg)


def check_ma30_cross(df: pd.DataFrame, value_name: str, indicator: str) -> None:
    if len(df) < 35:
        return
    d = df.copy()
    d["ma30"] = d[value_name].rolling(30).mean()

    latest = d.iloc[-1]
    prev = d.iloc[-2]

    if pd.isna(prev["ma30"]) or pd.isna(latest["ma30"]):
        return

    # Cruce alcista
    if prev[value_name] < prev["ma30"] and latest[value_name] > latest["ma30"]:
        msg = f"{indicator.upper()} CRUCE ALCISTA sobre MA30"
        log_line(ALERT_LOG_PATH, "SEÑAL: " + msg)
        notify_mac("Investment Lab — Señal", msg)

    # Cruce bajista
    if prev[value_name] > prev["ma30"] and latest[value_name] < latest["ma30"]:
        msg = f"{indicator.upper()} CRUCE BAJISTA bajo MA30"
        log_line(ALERT_LOG_PATH, "SEÑAL: " + msg)
        notify_mac("Investment Lab — Señal", msg)


def check_rate_change(df: pd.DataFrame, value_name: str, indicator: str) -> None:
    if len(df) < 2:
        return
    prev = float(df.iloc[-2][value_name])
    latest = float(df.iloc[-1][value_name])
    latest_date = df.iloc[-1]["fecha"]

    if latest != prev:
        direction = "SUBE" if latest > prev else "BAJA"
        msg = f"{indicator.upper()} {direction}: {prev:.2f}% → {latest:.2f}% ({latest_date})"
        log_line(ALERT_LOG_PATH, "SEÑAL: " + msg)
        notify_mac("Investment Lab — TPM cambió", msg)


# =========================
# SNAPSHOT
# =========================
def build_daily_snapshot(
    dolar_df: pd.DataFrame,
    uf_df: pd.DataFrame,
    tpm_df: pd.DataFrame,
    dxy_df: pd.DataFrame,
) -> pd.DataFrame:
    usd = dolar_df.copy()
    uf = uf_df.copy()
    tpm = tpm_df.copy()
    dxy = dxy_df.copy()

    usd["fecha"] = pd.to_datetime(usd["fecha"])
    uf["fecha"] = pd.to_datetime(uf["fecha"])
    tpm["fecha"] = pd.to_datetime(tpm["fecha"])
    dxy["fecha"] = pd.to_datetime(dxy["fecha"])

    usd = usd.sort_values("fecha")
    usd["usd_pct_change"] = usd["usd_clp"].pct_change() * 100
    usd["usd_ma30"] = usd["usd_clp"].rolling(30).mean()
    usd["usd_above_ma30"] = usd["usd_clp"] > usd["usd_ma30"]

    # Merge en base a fechas de USD
    df = (
        usd.merge(uf, on="fecha", how="left")
           .merge(tpm, on="fecha", how="left")
           .merge(dxy, on="fecha", how="left")
    )

    # Rellenos típicos
    df["uf"] = df["uf"].ffill()
    df["tpm"] = df["tpm"].ffill()
    df["dxy"] = df["dxy"].ffill()

    df.to_csv(SNAPSHOT_PATH, index=False, encoding="utf-8")
    return df


# =========================
# MAIN
# =========================
def main() -> None:
    run_log = LOG_DIR / "run.log"
    log_line(run_log, "---- RUN START ----")

    dfs: dict[str, pd.DataFrame] = {}
    statuses: dict[str, str] = {}

    # 1) Procesar indicadores
    for ind, cfg in INDICATORS.items():
        df = process_indicator(ind, cfg, run_log=run_log, statuses=statuses)
        dfs[ind] = df
        log_line(run_log, f"OK {ind}: {len(df)} filas (status={statuses.get(ind)})")

    # 2) Snapshot consolidado
    snapshot = build_daily_snapshot(dfs["dolar"], dfs["uf"], dfs["tpm"], dfs["dxy"])
    log_line(run_log, f"OK snapshot: {len(snapshot)} filas → {SNAPSHOT_PATH}")

    # 3) Señales/alertas
    if ALERTS["dolar_level"]["enabled"]:
        check_level(dfs["dolar"], "usd_clp", "dolar", ALERTS["dolar_level"]["threshold"])

    if ALERTS["dolar_shock"]["enabled"]:
        check_daily_change(dfs["dolar"], "usd_clp", "dolar", ALERTS["dolar_shock"]["pct_threshold"])

    if ALERTS["dolar_ma30_cross"]["enabled"]:
        check_ma30_cross(dfs["dolar"], "usd_clp", "dolar")

    if ALERTS["tpm_change"]["enabled"]:
        check_rate_change(dfs["tpm"], "tpm", "tpm")

    # 4) Resumen del run
    write_run_summary(dfs, statuses)
    log_line(run_log, f"OK run_summary: {DATA_DIR / 'run_summary.json'}")

    log_line(run_log, "---- RUN END ----")


if __name__ == "__main__":
    main()