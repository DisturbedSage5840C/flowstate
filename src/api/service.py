"""Query layer behind the Flow State UI (pure pandas, no web framework, so it is unit-testable).

Everything the UI shows comes from artifacts the pipeline already produced:

  * ``data/processed/train_real_large.parquet`` (falls back to ``train_real.parquet``): CPCB measurements joined to
    Sentinel-2 scenes, one row per (station, visit)
  * ``reports/real/screening_oof.parquet``: out-of-fold P(BOD > 3 mg/L) per visit (the PRIMARY model); a station's
    breach probability is the mean over its visits. Falls back to ``screening_shortlist.csv`` (older layout).
  * ``reports/real/spatial_knn_oof.parquet`` (optional): held-out spatial-KNN estimates per visit, exported by
    ``scripts/export_spatial_knn_oof.py``; ``reports/real/spatial_knn_summary.json`` holds its validation scores
  * ``reports/real/screening_metrics.json`` / ``dataset_summary*.json``: validation numbers and dataset facts

Nothing here invents data. Where the UI mock-up had content the backend has no source for (sewage outlets, STPs,
drainage networks, land use, forecasts) the API simply does not offer it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.city_proximity import CITIES

ROOT = Path(__file__).resolve().parents[2]

# --- risk banding ---------------------------------------------------------------------------------------------
# Screening basis: model breach probability P(BOD > 3 mg/L). Base rate is 29 %, so >= 0.5 means "the model thinks
# a breach is more likely than not" and 0.25-0.5 is around the base rate.
P_HIGH, P_MOD = 0.50, 0.25
# Measured basis (used per-year and for the single-parameter layers): thresholds on the measured value.
BOD_LIMIT, BOD_HIGH = 3.0, 6.0           # mg/L, CPCB Class B/C limit and "heavily polluted"
DO_LIMIT, DO_HIGH = 5.0, 4.0             # mg/L, CPCB Class B / C minimum
TURB_MOD, TURB_HIGH = 5.0, 25.0          # NTU, IS 10500 permissible limit; 25 = 5x the limit
TIER_BAND = {"Excellent": "low", "Good": "low", "Moderate": "mod", "Poor": "high", "Very Poor": "high"}
TREND_TOL = 0.10                          # +-10 % vs earlier visits counts as "flat"
WQI_CHANGE = 10.0                         # WQI points

INDICATORS = (("turbidity", "Turbidity", "NTU"), ("bod", "BOD", "mg/L"), ("do", "Dissolved oxygen", "mg/L"))


def _nan_to_none(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _band_bod(v):
    return None if v is None or pd.isna(v) else "high" if v > BOD_HIGH else "mod" if v > BOD_LIMIT else "low"


def _band_do(v):
    return None if v is None or pd.isna(v) else "high" if v < DO_HIGH else "mod" if v < DO_LIMIT else "low"


def _band_turb(v):
    return None if v is None or pd.isna(v) else "high" if v > TURB_HIGH else "mod" if v > TURB_MOD else "low"


def _band_prob(p):
    return None if p is None or pd.isna(p) else "high" if p >= P_HIGH else "mod" if p >= P_MOD else "low"


def _trend(latest, earlier: pd.Series) -> str:
    earlier = earlier.dropna()
    if pd.isna(latest) or earlier.empty:
        return "flat"
    base = earlier.mean()
    if base == 0:
        return "flat"
    change = (latest - base) / abs(base)
    return "up" if change > TREND_TOL else "down" if change < -TREND_TOL else "flat"


def _site_id(site: str) -> str:
    return hashlib.sha1(site.encode("utf-8")).hexdigest()[:10]


def _fmt(v, nd=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def _title(name: str) -> str:
    """CPCB station names are shouted and carry trailing noise; make them readable without altering meaning."""
    name = re.sub(r"\s+", " ", str(name)).strip(" ()")
    return name.title() if name.isupper() else name


class Store:
    """Loads the artifacts once and answers UI queries."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        proc = self.root / "data" / "processed"
        path = proc / "train_real_large.parquet"
        if not path.exists():
            path = proc / "train_real.parquet"
        if not path.exists():
            raise FileNotFoundError("No real training table found; run the real-data pipeline first (see README).")
        self.table_name = path.name
        t = pd.read_parquet(path)
        t["date"] = pd.to_datetime(t["date"])
        t["year"] = t["date"].dt.year
        t = t.sort_values(["site", "date"]).reset_index(drop=True)
        t["sid"] = t["site"].map({s: _site_id(s) for s in t["site"].unique()})
        self.table = t

        real = self.root / "reports" / "real"

        def read_json(name):
            p = real / name
            return json.loads(p.read_text()) if p.exists() else {}

        self.screening = read_json("screening_metrics.json")
        self.dataset = read_json("dataset_summary_large.json") or read_json("dataset_summary.json")
        self._prob = self._load_probabilities(real)

        oof = real / "spatial_knn_oof.parquet"
        self.knn_oof = pd.read_parquet(oof) if oof.exists() else None
        if self.knn_oof is not None:
            self.knn_oof["date"] = pd.to_datetime(self.knn_oof["date"])
        self.knn_summary = read_json("spatial_knn_summary.json")

        self.stations = self._build_stations()
        self._by_id = self.stations.set_index("id", drop=False)
        self._year_cache: dict[int, pd.DataFrame] = {}
        self.years = sorted(int(y) for y in t["year"].unique())

    @staticmethod
    def _load_probabilities(real: Path) -> dict:
        oof = real / "screening_oof.parquet"
        if oof.exists():
            o = pd.read_parquet(oof)
            return o.groupby("site")["bod_gt_3_proba"].mean().dropna().to_dict()
        sl = real / "screening_shortlist.csv"
        if sl.exists():
            return pd.read_csv(sl).drop_duplicates("site").set_index("site")["breach_probability"].to_dict()
        return {}

    # ------------------------------------------------------------------ station table
    def _build_stations(self) -> pd.DataFrame:
        t = self.table
        rows = []
        # nearest city name (the model features only store the distance)
        clat = np.array([c[1] for c in CITIES])
        clon = np.array([c[2] for c in CITIES])
        for site, g in t.groupby("site", sort=False):
            last = g.iloc[-1]
            earlier = g.iloc[:-1]
            p = self._prob.get(site)
            # station-level current state = latest visit, but keep the last non-null value of each indicator so a
            # visit that skipped one parameter does not blank the panel
            latest = {c: (g[c].dropna().iloc[-1] if g[c].notna().any() else np.nan)
                      for c in ("do", "bod", "turbidity", "wqi", "ph", "conductivity", "total_coliform")}
            wqi_change = None
            if len(g) >= 2 and pd.notna(last["wqi"]) and earlier["wqi"].notna().any():
                d = last["wqi"] - earlier["wqi"].mean()
                wqi_change = "worse" if d > WQI_CHANGE else "better" if d < -WQI_CHANGE else "stable"
            band = _band_prob(p) or TIER_BAND.get(last["wqi_tier"])
            d2 = np.hypot(clat - last["lat"], (clon - last["lon"]) * np.cos(np.radians(last["lat"]))) * 111.0
            ci = int(np.argmin(d2))
            rows.append({
                "id": last["sid"], "site": site, "name": _title(site), "state": last["state"],
                "lat": float(last["lat"]), "lon": float(last["lon"]), "type": last["water_body_type"],
                "visits": len(g), "first_date": g["date"].iloc[0], "last_date": last["date"],
                "prob": None if p is None else float(p),
                "basis": "screening" if p is not None else "measured",
                "risk": band or "low", "has_risk": band is not None,
                "bod": _nan_to_none(latest["bod"]), "do": _nan_to_none(latest["do"]),
                "turbidity": _nan_to_none(latest["turbidity"]), "wqi": _nan_to_none(latest["wqi"]),
                "tier": last["wqi_tier"] if isinstance(last["wqi_tier"], str) else None,
                "cpcb_class": last["cpcb_class"] if isinstance(last["cpcb_class"], str) else None,
                "change": wqi_change or "stable",
                "dist_city_km": _nan_to_none(last["dist_nearest_city_km"]),
                "urban_load": _nan_to_none(last["urban_load_index"]),
                "city": CITIES[ci][0],
                "scene_id": last["scene_id"], "scene_date": last["scene_date"], "scene_cloud": _nan_to_none(last["scene_cloud"]),
            })
        df = pd.DataFrame(rows)
        df["rb"] = df["bod"].map(_band_bod)
        df["rd"] = df["do"].map(_band_do)
        df["rt"] = df["turbidity"].map(_band_turb)
        df["breach_visits"] = df["site"].map(t[t["bod"].notna()].groupby("site")["bod"].apply(lambda s: int((s > BOD_LIMIT).sum())))
        df["bod_visits"] = df["site"].map(t[t["bod"].notna()].groupby("site").size())
        # percentile rank of breach probability (1.0 = most likely to breach)
        df["prob_rank"] = df["prob"].rank(pct=True)
        return df

    def _year_frame(self, year: int) -> pd.DataFrame:
        """Per-station measured state within one calendar year (mean of that year's visits)."""
        if year not in self._year_cache:
            g = self.table[self.table["year"] == year].groupby("sid")
            f = g.agg(wqi=("wqi", "mean"), bod=("bod", "mean"), do=("do", "mean"), turbidity=("turbidity", "mean"),
                      n=("wqi", "size"))
            f["tier"] = None
            self._year_cache[year] = f
        return self._year_cache[year]

    @staticmethod
    def _wqi_band(w):
        return None if pd.isna(w) else "high" if w >= 60 else "mod" if w >= 40 else "low"

    # ------------------------------------------------------------------ public queries
    def list_stations(self, year: int | None = None, state: str | None = None) -> list[dict]:
        s = self.stations
        if state:
            s = s[s["state"] == state]
        if year is None:
            out = s
            return [{
                "id": r.id, "name": r.name, "state": r.state, "lat": r.lat, "lon": r.lon, "type": r.type,
                "risk": r.risk, "p": None if pd.isna(r.prob) else round(r.prob, 3),
                "rb": r.rb, "rd": r.rd, "rt": r.rt, "ch": r.change,
            } for r in out.itertuples()]
        yf = self._year_frame(year)
        out = []
        for r in s.itertuples():
            if r.id not in yf.index:
                continue
            y = yf.loc[r.id]
            risk = self._wqi_band(y["wqi"])
            if risk is None:
                continue
            out.append({
                "id": r.id, "name": r.name, "state": r.state, "lat": r.lat, "lon": r.lon, "type": r.type,
                "risk": risk, "p": None, "rb": _band_bod(y["bod"]), "rd": _band_do(y["do"]), "rt": _band_turb(y["turbidity"]),
                "ch": "stable",
            })
        return out

    def search(self, q: str, limit: int = 8) -> list[dict]:
        q = q.strip().lower()
        if len(q) < 2:
            return []
        s = self.stations
        hit = s[s["site"].str.lower().str.contains(re.escape(q)) | s["state"].str.lower().str.contains(re.escape(q))
                | s["city"].str.lower().str.contains(re.escape(q))]
        hit = hit.sort_values("prob", ascending=False, na_position="last").head(limit)
        return [{"id": r.id, "name": r.name, "state": r.state, "lat": r.lat, "lon": r.lon, "risk": r.risk} for r in hit.itertuples()]

    def _row(self, sid: str):
        if sid not in self._by_id.index:
            raise KeyError(sid)
        return self._by_id.loc[sid]

    def _factors_and_signals(self, r) -> tuple[list[dict], list[str]]:
        signals: list[str] = []
        factors: list[dict] = []
        if pd.notna(r["bod"]) and r["bod"] > BOD_LIMIT:
            signals.append(f"Latest measured BOD {r['bod']:.1f} mg/L exceeds the CPCB {BOD_LIMIT:g} mg/L limit")
        if pd.notna(r["do"]) and r["do"] < DO_LIMIT:
            signals.append(f"Latest measured DO {r['do']:.1f} mg/L is below the CPCB Class B minimum ({DO_LIMIT:g} mg/L)")
        if r["bod_visits"] and r["bod_visits"] >= 2 and r["breach_visits"]:
            signals.append(f"BOD breached the limit on {int(r['breach_visits'])} of {int(r['bod_visits'])} recorded visits")
            factors.append({"hard": bool(r["breach_visits"] / r["bod_visits"] >= 0.5),
                            "text": f"BOD over limit on {int(r['breach_visits'])}/{int(r['bod_visits'])} visits"})
        if r["prob"] is not None and not pd.isna(r["prob"]):
            top = int(round((1 - r["prob_rank"]) * 100)) or 1
            signals.append(f"Screening model: {r['prob']:.0%} probability of breaching BOD {BOD_LIMIT:g} mg/L (top {top}% of stations)")
            factors.append({"hard": bool(r["prob"] >= P_HIGH), "text": f"Model breach probability {r['prob']:.0%}"})
        if r["dist_city_km"] is not None and not pd.isna(r["dist_city_km"]):
            near = r["dist_city_km"] <= 25
            if near:
                signals.append(f"{r['dist_city_km']:.0f} km from {r['city']} (urban load index {_fmt(r['urban_load'])})")
            factors.append({"hard": bool(near), "text": f"{r['dist_city_km']:.0f} km from {r['city']}"})
        if r["change"] == "worse":
            signals.append("Water-quality index worsened versus this station's earlier visits")
            factors.append({"hard": True, "text": "WQI worse than earlier visits"})
        elif r["change"] == "better":
            factors.append({"hard": False, "text": "WQI better than earlier visits"})
        if r["type"] and r["type"] != "unknown":
            factors.append({"hard": False, "text": f"Water body type: {r['type']}"})
        if not signals:
            signals.append("No measured limit breach or elevated model probability on record")
        return factors, signals

    def _recommendations(self, r) -> list[dict]:
        recs = []
        breach = pd.notna(r["bod"]) and r["bod"] > BOD_LIMIT
        near = r["dist_city_km"] is not None and not pd.isna(r["dist_city_km"]) and r["dist_city_km"] <= 25
        if near and (breach or r["risk"] == "high"):
            recs.append(("INVESTIGATE DISCHARGE",
                         f"Elevated risk within {r['dist_city_km']:.0f} km of {r['city']}, where sewage and industrial discharge concentrate.",
                         "Suggested action: field survey of drains and outfalls upstream of this station."))
        if r["risk"] == "high" and not breach:
            recs.append(("CONFIRM WITH LAB SAMPLING",
                         "The model flags this station but the latest measured BOD does not exceed the limit.",
                         "Suggested action: collect a fresh BOD sample before committing resources."))
        if r["change"] == "worse":
            recs.append(("MONITOR CLOSELY", "Water-quality index has worsened versus earlier visits.",
                         "Suggested action: increase sampling frequency to establish whether this persists."))
        if r["visits"] < 3:
            recs.append(("EXTEND RECORD", f"Only {int(r['visits'])} visit(s) are on record for this station.",
                         "Suggested action: more visits are needed before reading a trend."))
        if not recs:
            recs.append(("ROUTINE MONITORING", "No breach or adverse trend on record.", "Suggested action: continue scheduled monitoring."))
        return [{"n": f"{i + 1:02d}", "title": t, "evidence": e, "action": a} for i, (t, e, a) in enumerate(recs)]

    def station(self, sid: str) -> dict:
        r = self._row(sid)
        g = self.table[self.table["sid"] == sid]
        last = g.iloc[-1]
        earlier = g.iloc[:-1]
        readings = []
        for col, label, unit in INDICATORS:
            v = r[col]
            readings.append({
                "name": label, "unit": unit, "value": None if pd.isna(v) else round(float(v), 2),
                "trend": _trend(v, earlier[col]), "band": {"bod": r["rb"], "do": r["rd"], "turbidity": r["rt"]}[col],
                "n": int(g[col].notna().sum()),
            })
        factors, signals = self._factors_and_signals(r)
        yearly = []
        estimates = self._estimates(sid)
        for y, gy in g.groupby("year"):
            yearly.append({"year": int(y), "visits": len(gy), "risk": self._wqi_band(gy["wqi"].mean()),
                           "wqi": _nan_to_none(gy["wqi"].mean()), "bod": _nan_to_none(gy["bod"].mean()),
                           "do": _nan_to_none(gy["do"].mean()), "turbidity": _nan_to_none(gy["turbidity"].mean())})
        around = []
        if r["dist_city_km"] is not None and not pd.isna(r["dist_city_km"]):
            around.append({"dist": f"{r['dist_city_km']:.0f} km", "kind": f"Nearest major city: {r['city']}", "icon": "built"})
        if r["urban_load"] is not None and not pd.isna(r["urban_load"]):
            level = "high" if r["urban_load"] > 12 else "moderate" if r["urban_load"] > 9 else "low"
            around.append({"dist": f"{r['urban_load']:.1f}", "kind": f"Urban load index ({level})", "icon": "ind"})
        if r["prob"] is not None and not pd.isna(r["prob"]):
            rationale = (f"The screening model ranks this station in the top {int(round((1 - r['prob_rank']) * 100)) or 1}% for "
                         f"breaching BOD {BOD_LIMIT:g} mg/L (probability {r['prob']:.0%}). It is a triage ranking validated with "
                         f"site-blocked cross-validation, not a lab measurement.")
        else:
            rationale = "No screening probability for this station; risk band comes from the measured water-quality index."
        return {
            "id": sid, "name": r["name"], "raw_name": r["site"], "state": r["state"], "lat": r["lat"], "lon": r["lon"],
            "type": r["type"], "risk": r["risk"], "basis": r["basis"],
            "prob": None if r["prob"] is None or pd.isna(r["prob"]) else round(float(r["prob"]), 3),
            "position": f"{r['state']} · {r['type'] if r['type'] != 'unknown' else 'surface water'} · nearest city {r['city']} ({_fmt(r['dist_city_km'], 0)} km)",
            "visits": int(r["visits"]), "first_date": r["first_date"].strftime("%d %b %Y"), "last_date": r["last_date"].strftime("%d %b %Y"),
            "readings": readings, "around": around, "rationale": rationale, "signals": signals, "factors": factors,
            "wqi": _nan_to_none(r["wqi"]), "tier": r["tier"], "cpcb_class": r["cpcb_class"],
            "yearly": yearly, "estimates": estimates, "recommendations": self._recommendations(r),
            "scene": {"id": last["scene_id"], "date": pd.Timestamp(last["scene_date"]).strftime("%d %b %Y") if pd.notna(last["scene_date"]) else None,
                      "cloud": _nan_to_none(last["scene_cloud"])},
        }

    def _estimates(self, sid: str) -> dict | None:
        """Held-out neighbour+satellite estimate next to the measured value, latest visit with an estimate.

        Every estimate is out-of-fold: this station's own record is excluded from the neighbours and from
        the model's training rows. ``skill`` carries the model's overall validation score per indicator."""
        if self.knn_oof is None:
            return None
        site = self._by_id.loc[sid]["site"]
        rows = self.knn_oof[self.knn_oof["site"] == site]
        if rows.empty:
            return None
        g = self.table[self.table["sid"] == sid].set_index("date")
        out = []
        for col, label, unit in INDICATORS:
            est_col = f"{col}_est"
            if est_col not in rows.columns:
                continue
            have = rows[rows[est_col].notna()].sort_values("date")
            if have.empty:
                continue
            r = have.iloc[-1]
            measured = g[col].get(r["date"]) if r["date"] in g.index else None
            if isinstance(measured, pd.Series):
                measured = measured.iloc[0]
            tgt = (self.knn_summary.get("targets") or {}).get(col) or {}
            combo = tgt.get("knn_plus_xgboost") or {}
            skill = combo.get("R2_log") if combo.get("R2_log") == combo.get("R2_log") and combo.get("R2_log") is not None else combo.get("R2")
            out.append({
                "name": label, "unit": unit, "date": pd.Timestamp(r["date"]).strftime("%d %b %Y"),
                "estimate": round(float(r[est_col]), 2), "measured": _nan_to_none(measured),
                "nearest_km": _nan_to_none(r.get(f"{col}_nearest_km")),
                "skill": None if skill is None else round(float(skill), 3),
                "skill_metric": "R2_log" if col in ("bod", "turbidity") else "R2",
                "spearman": _nan_to_none(combo.get("spearman")),
            })
        return {"items": out} if out else None

    def compare(self, a: str, b: str) -> dict:
        ra, rb = self._row(a), self._row(b)
        dist = float(np.hypot(ra["lat"] - rb["lat"], (ra["lon"] - rb["lon"]) * np.cos(np.radians(ra["lat"]))) * 111.0)

        def col(r):
            return {"name": r["name"], "risk": r["risk"], "prob": r["prob"], "bod": r["bod"], "do": r["do"],
                    "turbidity": r["turbidity"], "wqi": r["wqi"], "dist_city_km": r["dist_city_km"],
                    "city": r["city"], "visits": int(r["visits"])}

        return {"a": {k: _nan_to_none(v) for k, v in col(ra).items()},
                "b": {k: _nan_to_none(v) for k, v in col(rb).items()}, "distance_km": round(dist, 1)}

    def overview(self, state: str | None = None) -> dict:
        s = self.stations if not state else self.stations[self.stations["state"] == state]
        m = self.screening.get("targets", {}).get("bod_gt_3", {})
        # metrics layout changed between pipeline versions: auc/roc_auc, "10%"/"top_10pct"
        p10 = (m.get("precision_at_k") or {}).get("10%", (m.get("precision_at_k") or {}).get("top_10pct"))
        base = m.get("base_rate")
        latest_bod = s["bod"].dropna()
        return {
            "scope": state or "India",
            "stations": int(len(s)),
            "visits": int(len(self.table)) if not state else int((self.table["state"] == state).sum()),
            "high": int((s["risk"] == "high").sum()), "moderate": int((s["risk"] == "mod").sum()),
            "low": int((s["risk"] == "low").sum()),
            "confirmed_breaches": int((latest_bod > BOD_LIMIT).sum()),
            "deteriorating": int((s["change"] == "worse").sum()),
            "states": sorted(self.stations["state"].unique().tolist()),
            "years": self.years,
            "date_range": [self.table["date"].min().strftime("%d %b %Y"), self.table["date"].max().strftime("%d %b %Y")],
            "model": {
                "target": "BOD > 3 mg/L (CPCB Class B/C limit)",
                "roc_auc": m.get("auc", m.get("roc_auc")), "base_rate": base,
                "precision_top_10pct": p10,
                "lift_top_10pct": (p10 / base) if p10 and base else None,
                "validation": self.screening.get("validation", "5-fold site-blocked spatial CV, out-of-fold probabilities"),
                "n": m.get("n"),
            },
            "thresholds": {"high": P_HIGH, "mod": P_MOD},
            "table": self.table_name,
        }

    def area_recommendations(self) -> list[dict]:
        s = self.stations[self.stations["risk"] == "high"]
        top = s.sort_values("prob", ascending=False).head(50)
        near = int((top["dist_city_km"] <= 25).sum())
        model_only = int(((s["bod"].isna()) | (s["bod"] <= BOD_LIMIT)).sum())
        worse = int((self.stations["change"] == "worse").sum())
        return [
            {"n": "01", "title": "INVESTIGATE DISCHARGE",
             "evidence": f"{near} of the 50 highest-risk stations sit within 25 km of a major city (distance to cities is one of the model's inputs).",
             "action": "Suggested action: prioritise field surveys of drains and outfalls upstream of these stations."},
            {"n": "02", "title": "CONFIRM WITH LAB SAMPLING",
             "evidence": f"{model_only} of {len(s)} high-risk stations have no measured BOD breach on their latest record; the flag is model-only.",
             "action": "Suggested action: sample before committing remediation resources."},
            {"n": "03", "title": "MONITOR DETERIORATION",
             "evidence": f"{worse} stations show a worse water-quality index than their earlier visits.",
             "action": "Suggested action: raise sampling frequency at these stations."},
        ]

    def priorities(self, n: int = 8, state: str | None = None) -> list[dict]:
        s = self.stations[self.stations["prob"].notna()]
        if state:
            s = s[s["state"] == state]
        s = s.sort_values("prob", ascending=False).head(n)
        out = []
        for r in s.itertuples():
            bod = f" · latest BOD {r.bod:.1f} mg/L" if r.bod is not None and not pd.isna(r.bod) else ""
            out.append({"id": r.id, "name": f"{r.name} — {r.state}", "risk": r.risk, "lat": r.lat, "lon": r.lon,
                        "reason": f"{r.prob:.0%} breach probability{bod}"})
        return out

    def alerts(self, n: int = 8) -> list[dict]:
        """Stations whose most recent visit crossed the BOD limit or worsened by a WQI tier versus the visit before."""
        out = []
        for sid, g in self.table.groupby("sid", sort=False):
            if len(g) < 2:
                continue
            last, prev = g.iloc[-1], g.iloc[-2]
            st = self._by_id.loc[sid]
            crossed = pd.notna(last["bod"]) and pd.notna(prev["bod"]) and prev["bod"] <= BOD_LIMIT < last["bod"]
            worsened = (pd.notna(last["wqi"]) and pd.notna(prev["wqi"]) and last["wqi"] - prev["wqi"] >= 20
                        and last["wqi"] >= 60)
            if not (crossed or worsened):
                continue
            if crossed:
                title, sev = "BOD LIMIT CROSSED", "high" if last["bod"] > BOD_HIGH else "mod"
                ev = f"BOD rose from {prev['bod']:.1f} to {last['bod']:.1f} mg/L (limit {BOD_LIMIT:g})."
            else:
                title, sev = "DETERIORATION ALERT", "high" if last["wqi"] >= 80 else "mod"
                ev = f"Water-quality index rose from {prev['wqi']:.0f} to {last['wqi']:.0f} (higher is worse)."
            out.append({"id": sid, "sev": sev, "title": title, "loc": f"{st['name']} — {st['state']}",
                        "date": last["date"].strftime("%d %b %Y"), "_d": last["date"], "evidence": ev,
                        "lat": float(st["lat"]), "lon": float(st["lon"])})
        out.sort(key=lambda a: a["_d"], reverse=True)
        for a in out:
            a.pop("_d")
        return out[:n]

    def cities(self, min_stations: int = 4, radius_km: float = 50) -> list[dict]:
        s = self.stations
        out = []
        for name, lat, lon, pop in CITIES:
            d = np.hypot(s["lat"] - lat, (s["lon"] - lon) * np.cos(np.radians(lat))) * 111.0
            near = s[d <= radius_km]
            if len(near) < min_stations:
                continue
            share = float((near["risk"] == "high").mean())
            out.append({"name": name, "lat": lat, "lon": lon, "stations": int(len(near)), "high_share": round(share, 3),
                        "risk": "high" if share >= 0.3 else "mod" if share >= 0.12 else "low"})
        out.sort(key=lambda c: c["stations"], reverse=True)
        return out

    def scene_near(self, lat: float, lon: float, max_km: float = 60) -> dict | None:
        s = self.stations
        d = np.hypot(s["lat"] - lat, (s["lon"] - lon) * np.cos(np.radians(lat))) * 111.0
        i = int(np.argmin(d.values))
        if d.iloc[i] > max_km:
            return None
        r = s.iloc[i]
        return {"station": r["name"], "distance_km": round(float(d.iloc[i]), 1), "scene_id": r["scene_id"],
                "date": pd.Timestamp(r["scene_date"]).strftime("%d %b %Y") if pd.notna(r["scene_date"]) else None,
                "cloud": _nan_to_none(r["scene_cloud"])}

    # ------------------------------------------------------------------ Ask Flow State
    def ask(self, query: str) -> dict:
        """Deterministic keyword parser over the station table (no LLM). Says plainly what it understood."""
        q = query.lower()
        s = self.stations.copy()
        parts: list[str] = []
        notes: list[str] = []

        states = [st for st in self.stations["state"].unique() if st.lower().replace("&", "and") in q.replace("&", "and")]
        if states:
            s = s[s["state"].isin(states)]
            parts.append(", ".join(states))
        cities = [c for c in CITIES if re.search(rf"\b{re.escape(c[0].lower())}\b", q)]
        m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*km", q)
        radius = float(m.group(1)) if m else None
        if cities:
            c = cities[0]
            rad = radius if radius and not re.search(r"sewage|drain|outlet", q) else 50.0
            d = np.hypot(s["lat"] - c[1], (s["lon"] - c[2]) * np.cos(np.radians(c[1]))) * 111.0
            s = s[d <= rad]
            parts.append(f"within {rad:g} km of {c[0]}")
        elif radius is not None:
            if re.search(r"sewage|drain|outlet|stp|treatment", q):
                notes.append("There is no sewage-outlet or drain inventory in the data; using distance to the nearest "
                             "major city as a coarse proxy for urban discharge.")
            s = s[s["dist_city_km"] <= radius]
            parts.append(f"within {radius:g} km of a major city")

        if re.search(r"hotspot|polluted|worst|highest|high[- ]risk|major", q):
            s = s[s["risk"] == "high"]
            parts.append("high screening risk")
        if re.search(r"deteriorat|worse|worsen|declin", q):
            s = s[s["change"] == "worse"]
            parts.append("water quality worse than earlier visits")
        if re.search(r"\bbod\b|oxygen demand", q):
            s = s[s["bod"] > BOD_LIMIT]
            parts.append(f"latest BOD > {BOD_LIMIT:g} mg/L")
        if re.search(r"dissolved oxygen|\bdo\b|low oxygen", q):
            s = s[s["do"] < DO_LIMIT]
            parts.append(f"latest DO < {DO_LIMIT:g} mg/L")
        if re.search(r"turbid", q):
            s = s[s["turbidity"] > TURB_HIGH]
            parts.append(f"latest turbidity > {TURB_HIGH:g} NTU")
        for wt in ("lake", "river", "reservoir"):
            if re.search(rf"\b{wt}s?\b", q):
                s = s[s["type"] == wt]
                parts.append(wt + "s")

        if not parts:
            return {"understood": False, "count": 0, "ids": [], "label": "",
                    "summary": "I could not map that to a filter. Try a state or city, \"high-risk\", \"deteriorated\", "
                               "\"BOD\", \"low oxygen\", \"turbidity\", \"lakes\" or \"within 25 km of a city\".",
                    "notes": notes}
        label = " · ".join(parts)
        summary = (f"{len(s)} station{'s' if len(s) != 1 else ''} match: {label}. This reflects statistical association in CPCB "
                   f"measurements and the screening model, for investigation, not a causal claim.")
        return {"understood": True, "count": int(len(s)), "ids": s["id"].tolist(), "label": label, "summary": summary,
                "notes": notes}
