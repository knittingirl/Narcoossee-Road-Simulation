#!/usr/bin/env python3
import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import requests

from runner_network_flexible import (
    apply_intersection_overrides,
    build_network_config_from_feeder_cosites,
    monte_carlo_network,
)
from network_data import FEEDER_INTERSECTIONS, CORRIDOR


HISTORICAL_AADT_URL = (
    "https://services1.arcgis.com/O1JpcwDW8sjYuddV/ArcGIS/rest/services/"
    "Annual_Average_Daily_Traffic_Historical_TDA/FeatureServer/0/query"
)
CACHE_FILE = "fdot_historical_cache.json"
TIMEOUT = (5, 20)


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


CACHE = load_cache()


def _query_historical(where: str) -> list[dict]:
    params = {
        "f": "json",
        "where": where,
        "returnGeometry": "false",
        "orderByFields": "YEAR_ ASC",
        "outFields": (
            "YEAR_,COSITE,ROADWAY,COUNTY,DESC_FRM,DESC_TO,AADT,"
            "AADTFLG,KFCTR,KFLG,DFCTR,DFLG,TFCTR,TFLG"
        ),
    }
    response = requests.get(HISTORICAL_AADT_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"FDOT historical query failed: {payload['error']}")
    return [feature.get("attributes", {}) for feature in payload.get("features", [])]


def fetch_historical_by_cosite(cosite: str, *, use_cache: bool = True) -> list[dict]:
    cosite = str(cosite).strip()
    cache_key = f"cosite::{cosite}"
    if use_cache and cache_key in CACHE:
        print(f"Using cached historical data for COSITE {cosite}")
        return CACHE[cache_key]

    print(f"Fetching historical data for COSITE {cosite}")
    safe_cosite = cosite.replace("'", "''")
    rows = _query_historical(f"COSITE = '{safe_cosite}'")
    CACHE[cache_key] = rows
    save_cache(CACHE)
    return rows


def build_historical_table(feeders) -> pd.DataFrame:
    rows = []

    for feeder in feeders:
        cosite = str(getattr(feeder, "cosite", "") or "").strip()
        name = getattr(feeder, "name", None) or f"COSITE {cosite}"
        node_id = getattr(feeder, "node_id", None)
        if not cosite:
            continue

        for attrs in fetch_historical_by_cosite(cosite):
            rows.append({
                "year": attrs.get("YEAR_"),
                "cosite": attrs.get("COSITE") or cosite,
                "node_id": node_id,
                "intersection_name": name,
                "roadway": attrs.get("ROADWAY"),
                "county": attrs.get("COUNTY"),
                "desc_from": attrs.get("DESC_FRM"),
                "desc_to": attrs.get("DESC_TO"),
                "aadt": attrs.get("AADT"),
                "k_factor": attrs.get("KFCTR"),
                "d_factor": attrs.get("DFCTR"),
                "truck_factor": attrs.get("TFCTR"),
                "aadt_flag": attrs.get("AADTFLG"),
                "k_flag": attrs.get("KFLG"),
                "d_flag": attrs.get("DFLG"),
                "truck_flag": attrs.get("TFLG"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ["aadt", "k_factor", "d_factor", "truck_factor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["year", "intersection_name", "cosite"]).reset_index(drop=True)
    return df

def normalize_percentage(value):
    if value is None:
        return default
    v = float(value)
    if v > 1.0:
        v = v / 100.0
    return v

def build_year_overrides(historical_df: pd.DataFrame, year: int) -> dict[str, dict]:
    year_df = historical_df[historical_df["year"] == year].copy()
    overrides: dict[str, dict] = {}

    for row in year_df.itertuples(index=False):
        node_id = getattr(row, "node_id", None)
        if not node_id:
            continue

        patch = {}
        if pd.notna(row.aadt):
            patch["side_seed.aadt"] = float(row.aadt)
        if pd.notna(row.k_factor):
            patch["side_seed.k_factor"] = normalize_percentage(float(row.k_factor))
        if pd.notna(row.d_factor):
            patch["side_seed.directional_factor"] = normalize_percentage(float(row.d_factor))
        if pd.notna(row.truck_factor):
            patch["side_seed.truck_factor"] = normalize_percentage(float(row.truck_factor))

        if patch:
            overrides[str(node_id)] = patch

    return overrides


def available_years(historical_df: pd.DataFrame) -> list[int]:
    valid = historical_df.dropna(subset=["year", "node_id"])
    if valid.empty:
        return []

    feeder_count = valid["node_id"].nunique()
    counts = valid.groupby("year")["node_id"].nunique()
    years = counts[counts == feeder_count].index.tolist()
    return [int(year) for year in years]


def main() -> None:
    base = build_network_config_from_feeder_cosites(
        corridor=CORRIDOR,
        feeders=FEEDER_INTERSECTIONS,
    )

    historical_df = build_historical_table(FEEDER_INTERSECTIONS)
    if historical_df.empty:
        raise ValueError("No historical FDOT rows were returned for the configured FEEDER_INTERSECTIONS.")

    years = available_years(historical_df)
    if not years:
        raise ValueError("No year had complete historical coverage for all configured FEEDER_INTERSECTIONS.")

    runs = 20
    rows = []

    for year in years:
        cfg = apply_intersection_overrides(base, build_year_overrides(historical_df, year))
        res = monte_carlo_network(cfg, runs=runs)
        rows.append({
            "year": year,
            "mean_queue": res["mean_queue"],
            "mean_delay": res["mean_delay"],
            "p50_tt": res["p50_tt"],
            "p90_tt": res["p90_tt"],
            "spillback_freq": res["spillback_freq"],
            "mainline_spillback_freq": res["mainline_spillback_freq"],
            "side_spillback_freq": res["side_spillback_freq"],
            "mainline_spill_extent": res["mainline_spill_extent"],
            "side_spill_extent": res["side_spill_extent"],
            "network_spill_extent": res["network_spill_extent"],
            "total_entered_veh": res["total_entered_veh"],
            "total_exited_veh": res["total_exited_veh"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("runner_lines_historical_results.csv", index=False)
    historical_df.to_csv("runner_lines_historical_raw.csv", index=False)
    print(df)

    plt.figure()
    plt.plot(df["year"], df["mean_queue"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean Queue (veh)")
    plt.title("Historical Sensitivity: Mean Queue")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_historical_queue.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["year"], df["mean_delay"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean Delay (s/veh)")
    plt.title("Historical Sensitivity: Mean Delay")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_historical_delay.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["year"], df["p90_tt"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("P90 Travel Time (s)")
    plt.title("Historical Sensitivity: Reliability")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_historical_p90.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["year"], df["network_spill_extent"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Network Spill Extent")
    plt.title("Historical Sensitivity: Spillback Extent")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_historical_spill_extent.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["year"], df["total_exited_veh"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Total Exited Vehicles")
    plt.title("Historical Sensitivity: Throughput")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_historical_throughput.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
