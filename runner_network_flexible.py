#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
import os

from arrival_generator import PlatoonArrivalGenerator
from node_solver import godunov_node_solver
from postprocess import (
    compute_mean_delay,
    compute_mean_queue,
    mean_confidence_interval,
    spillback_frequency,
    travel_time_stats,
)
from queue_module import StopLineQueue
from simulation_framework import aadt_to_peak_rate, build_ctm

RCI_TOP30_URL = "https://gis.fdot.gov/arcgis/rest/services/RCI_Layers/FeatureServer/29/query"
AADT_QUERY_URL = "https://gis.fdot.gov/arcgis/rest/services/RCI_Layers/FeatureServer/0/query"
DEFAULT_CELL_LENGTH_M = 100.0
DEFAULT_RHO_JAM = 0.15
DEFAULT_TURNING_PROPORTIONS = np.array([[0.93, 0.07], [0.65, 0.35]], dtype=float)
DEFAULT_RED_MULTIPLIER = 0.05
DEFAULT_K_FACTOR = 0.10
DEFAULT_DIRECTIONAL_FACTOR = 0.60
DEFAULT_SAT_FLOW_PER_LANE = 1900.0 / 3600.0


@dataclass
class ApproachSeed:
    road_id: str | None = None
    road_name: str | None = None
    county: str = "Osceola"
    aadt: float = 10000.0
    lanes: int = 1
    speed_mph: float = 35.0
    length_m: float = 250.0
    truck_factor: float = 0.05
    directional_factor: float = 0.60


@dataclass
class SignalPlan:
    cycle_s: float = 120.0
    green_main_s: float = 70.0
    green_side_s: float = 35.0
    offset_s: float = 0.0
    red_multiplier: float = DEFAULT_RED_MULTIPLIER

    def green_fractions(self, t_s: float) -> tuple[float, float]:
        phase_t = (t_s - self.offset_s) % self.cycle_s
        main_green = 1.0 if phase_t < self.green_main_s else 0.0
        side_green = 1.0 if self.green_main_s <= phase_t < self.green_main_s + self.green_side_s else 0.0
        return main_green, side_green


@dataclass
class FeederIntersectionSpec:
    cosite: str
    node_id: str | None = None
    name: str | None = None
    signal: SignalPlan = field(default_factory=SignalPlan)
    notes: str = ""


@dataclass
class CorridorSpec:
    road_id: str = "92050000"
    county: str = "Osceola"


@dataclass
class IntersectionConfig:
    node_id: str
    name: str
    mainline: ApproachSeed
    side_street: ApproachSeed
    signal: SignalPlan = field(default_factory=SignalPlan)
    notes: str = ""


@dataclass
class BuiltIntersection:
    cfg: IntersectionConfig
    main_seed: dict[str, Any]
    side_seed: dict[str, Any]


@dataclass
class NetworkConfig:
    intersections: list[BuiltIntersection]
    dt_s: float = 2.0
    horizon_s: float = 3600.0
    peak_k_factor: float = DEFAULT_K_FACTOR
    default_directional_factor: float = DEFAULT_DIRECTIONAL_FACTOR
    burst_prob: float = 0.03
    mean_burst_size: float = 4.0
    intra_burst_headway_s: float = 1.0
    turning_proportions: np.ndarray = field(default_factory=lambda: DEFAULT_TURNING_PROPORTIONS.copy())


@dataclass
class LinkState:
    name: str
    params: dict[str, Any]
    engine: Any
    geometry: Any
    fd: Any

    @property
    def cell_length_m(self) -> float:
        return float(self.geometry.cell_length_m)

    @property
    def lanes(self) -> int:
        return int(self.params["lanes"])

    def receiving_head(self) -> float:
        return float(self.fd.receiving(np.array([self.engine.rho[0]]))[0])

    def tail_blocked(self) -> bool:
        return bool(self.engine.rho[-1] >= self.fd.rho_crit)

    def total_veh(self) -> float:
        return float(np.sum(self.engine.rho) * self.cell_length_m)

    def travel_time_s(self) -> float:
        rho = self.engine.rho
        q = np.minimum(self.fd.v_free * rho * self.fd.lanes, self.fd.flow_capacity())
        speed = np.where(rho > 1e-9, q / (rho * self.fd.lanes + 1e-9), self.fd.v_free)
        speed = np.clip(speed, 1.0, self.fd.v_free)
        return float(np.sum(self.cell_length_m / speed))

    def advance(self, upstream_inflow: float, downstream_capacity: float) -> np.ndarray:
        return self.engine.step(external_inflow=upstream_inflow, downstream_capacity=downstream_capacity)


def normalize_percentage(value: Any, default: float, lo: float, hi: float) -> float:
    if value is None:
        return default
    v = float(value)
    if v > 1.0:
        v = v / 100.0
    return max(lo, min(v, hi))


@lru_cache(maxsize=64)
def _query_rci(where: str) -> list[dict[str, Any]]:
    params = {
        "f": "json",
        "where": where,
        "returnGeometry": "false",
        "outFields": (
            "ROADWAY,STROADNO,COUNTY,SECTADT,NOLANES_R,NOLANES_L,"
            "MAXSPEED_R,MAXSPEED_L,AVGTFACT,AVGDFACT,Shape_Leng,Shape__Length"
        ),
    }
    r = requests.get(RCI_TOP30_URL, params=params, timeout=(2, 4))
    r.raise_for_status()
    return [f["attributes"] for f in r.json().get("features", [])]


@lru_cache(maxsize=128)
def _query_aadt(where: str) -> list[dict[str, Any]]:
    params = {
        "f": "json",
        "where": where,
        "returnGeometry": "false",
        "outFields": (
            "COSITE,ROADWAY,COUNTY,AADT,KFCTR,DFCTR,BEGIN_POST,END_POST,"
            "Shape__Length,SHAPE_LENG,DESC_FRM,DESC_TO,YEAR_"
        ),
    }
    r = requests.get(AADT_QUERY_URL, params=params, timeout=(2, 4))
    r.raise_for_status()
    return [f["attributes"] for f in r.json().get("features", [])]


def _best_count_site_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No FDOT count-site rows found")
    return max(
        rows,
        key=lambda attrs: (
            attrs.get("AADT") or 0,
            attrs.get("YEAR_") or 0,
            attrs.get("Shape__Length") or attrs.get("SHAPE_LENG") or 0,
        ),
    )


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No FDOT rows found")
    return max(
        rows,
        key=lambda attrs: (
            attrs.get("SECTADT") or 0,
            attrs.get("NOLANES_R") or 0,
            attrs.get("NOLANES_L") or 0,
            attrs.get("MAXSPEED_R") or attrs.get("MAXSPEED_L") or 0,
            attrs.get("Shape__Length") or attrs.get("Shape_Leng") or 0,
        ),
    )


def _row_to_seed(best: dict[str, Any], default_county: str) -> dict[str, Any]:
    lanes_r = best.get("NOLANES_R") or 0
    lanes_l = best.get("NOLANES_L") or 0
    return {
        "road_id": best.get("ROADWAY"),
        "road_name": best.get("STROADNO"),
        "county": best.get("COUNTY") or default_county,
        "aadt": float(best.get("SECTADT") or 0.0),
        "lanes": int(max(lanes_r, lanes_l, 1)),
        "speed_mph": float(best.get("MAXSPEED_R") or best.get("MAXSPEED_L") or 45.0),
        "length_m": float(best.get("Shape__Length") or best.get("Shape_Leng") or 350.0),
        "truck_factor": normalize_percentage(best.get("AVGTFACT"), 0.05, 0.0, 0.30),
        "directional_factor": normalize_percentage(best.get("AVGDFACT"), 0.60, 0.50, 0.70),
    }


@lru_cache(maxsize=64)
def fetch_best_segment_seed_by_id(road_id: str, county: str) -> dict[str, Any]:
    rows = _query_rci(f"ROADWAY = '{road_id}' AND UPPER(COUNTY) = '{county.upper()}'")
    return _row_to_seed(_best_row(rows), county)


@lru_cache(maxsize=64)
def fetch_best_segment_seed_by_name(road_name: str, county: str) -> dict[str, Any]:
    name = road_name.upper().replace("'", "''")
    rows = _query_rci(f"UPPER(STROADNO) = '{name}' AND UPPER(COUNTY) = '{county.upper()}'")
    return _row_to_seed(_best_row(rows), county)


@lru_cache(maxsize=128)
def fetch_count_site_by_cosite(cosite: str) -> dict[str, Any]:
    value = str(cosite).strip().replace("'", "''")
    rows = _query_aadt(f"COSITE = '{value}'")
    return _best_count_site_row(rows)

CACHE_FILE = "fdot_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

CACHE = load_cache()

def build_seed_from_cosite(cosite: str) -> dict[str, Any]:
    cosite = str(cosite)

    # checking if cosite is cached.
    if cosite in CACHE:
        print(f"Using cached data for COSITE {cosite}")
        return CACHE[cosite]

    print(f"Fetching API data for COSITE {cosite}")
    
    site = fetch_count_site_by_cosite(cosite)
    county = str(site.get("COUNTY") or "")
    roadway = str(site.get("ROADWAY") or "")
    if not roadway:
        raise ValueError(f"COSITE {cosite} did not return a ROADWAY value")

    seed = fetch_best_segment_seed_by_id(roadway, county)
    seed = dict(seed)

    aadt = site.get("AADT")
    if aadt is not None:
        seed["aadt"] = float(aadt)

    kfctr = site.get("KFCTR")
    if kfctr is not None:
        seed["k_factor"] = normalize_percentage(kfctr, DEFAULT_K_FACTOR, 0.01, 0.50)

    dfctr = site.get("DFCTR")
    if dfctr is not None:
        seed["directional_factor"] = normalize_percentage(dfctr, DEFAULT_DIRECTIONAL_FACTOR, 0.50, 0.99)

    site_length = site.get("Shape__Length") or site.get("SHAPE_LENG")
    if site_length:
        seed["length_m"] = float(site_length)

    seed["cosite"] = str(site.get("COSITE") or cosite)
    seed["count_site_year"] = site.get("YEAR_")
    seed["desc_from"] = site.get("DESC_FRM")
    seed["desc_to"] = site.get("DESC_TO")
    CACHE[cosite] = seed
    save_cache(CACHE)
    return seed


def intersection_from_feeder_cosite(
    cosite: str,
    corridor: CorridorSpec | None = None,
    *,
    node_id: str | None = None,
    name: str | None = None,
    signal: SignalPlan | None = None,
    notes: str = "",
) -> IntersectionConfig:
    corridor = corridor or CorridorSpec()
    side_seed = build_seed_from_cosite(cosite)
    main_seed = fetch_best_segment_seed_by_id(corridor.road_id, corridor.county)

    display_name = name or side_seed.get("road_name") or f"COSITE {cosite}"
    ident = node_id or str(display_name).strip().lower().replace(" ", "_").replace("-", "_")

    return IntersectionConfig(
        node_id=ident,
        name=str(display_name),
        mainline=ApproachSeed(**main_seed),
        side_street=ApproachSeed(**{k: v for k, v in side_seed.items() if k in ApproachSeed.__dataclass_fields__}),
        signal=signal or SignalPlan(),
        notes=notes,
    )


def build_network_config_from_feeder_cosites(
    feeders: list[str | FeederIntersectionSpec],
    corridor: CorridorSpec | None = None,
    dt_s: float = 2.0,
    horizon_s: float = 3600.0,
) -> NetworkConfig:
    corridor = corridor or CorridorSpec()
    intersections: list[IntersectionConfig] = []
    for idx, feeder in enumerate(feeders):
        if isinstance(feeder, FeederIntersectionSpec):
            intersections.append(
                intersection_from_feeder_cosite(
                    feeder.cosite,
                    corridor,
                    node_id=feeder.node_id,
                    name=feeder.name,
                    signal=feeder.signal,
                    notes=feeder.notes,
                )
            )
        else:
            intersections.append(
                intersection_from_feeder_cosite(
                    str(feeder),
                    corridor,
                    node_id=f"node_{idx+1}",
                    signal=SignalPlan(),
                )
            )
    return build_network_config(intersections=intersections, dt_s=dt_s, horizon_s=horizon_s)


def materialize_seed(seed: ApproachSeed) -> dict[str, Any]:
    base = {
        "road_id": seed.road_id,
        "road_name": seed.road_name,
        "county": seed.county,
        "aadt": float(seed.aadt),
        "lanes": int(seed.lanes),
        "speed_mph": float(seed.speed_mph),
        "length_m": float(seed.length_m),
        "truck_factor": float(seed.truck_factor),
        "directional_factor": float(seed.directional_factor),
    }
    try:
        if seed.road_id:
            return fetch_best_segment_seed_by_id(seed.road_id, seed.county)
        if seed.road_name:
            return fetch_best_segment_seed_by_name(seed.road_name, seed.county)
    except Exception:
        pass
    return base


def build_network_config(intersections: list[IntersectionConfig] | None = None, dt_s: float = 2.0, horizon_s: float = 3600.0) -> NetworkConfig:
    built = []
    
    for item in deepcopy(intersections if intersections is not None else DEFAULT_INTERSECTIONS):
        built.append(BuiltIntersection(cfg=item, main_seed=materialize_seed(item.mainline), side_seed=materialize_seed(item.side_street)))
   
    return NetworkConfig(intersections=built, dt_s=dt_s, horizon_s=horizon_s)


def build_default_config(dt_s: float = 2.0, horizon_s: float = 3600.0) -> NetworkConfig:
    return build_network_config(dt_s=dt_s, horizon_s=horizon_s)


def clone_config(cfg: NetworkConfig) -> NetworkConfig:
    return deepcopy(cfg)


def scale_all_mainline_demand(cfg: NetworkConfig, factor: float) -> NetworkConfig:
    out = clone_config(cfg)
    for node in out.intersections:
        node.main_seed["aadt"] *= factor
    return out


def add_lane_to_all_mainline_segments(cfg: NetworkConfig, n: int = 1) -> NetworkConfig:
    out = clone_config(cfg)
    for node in out.intersections:
        node.main_seed["lanes"] = int(node.main_seed["lanes"]) + n
    return out


def adjust_mainline_green(cfg: NetworkConfig, delta_s: float) -> NetworkConfig:
    out = clone_config(cfg)
    for node in out.intersections:
        sig = node.cfg.signal
        sig.green_main_s = min(max(sig.green_main_s + delta_s, 5.0), sig.cycle_s - 5.0)
    return out


def apply_intersection_overrides(cfg: NetworkConfig, overrides: dict[str, dict[str, Any]]) -> NetworkConfig:
    out = clone_config(cfg)
    for node in out.intersections:
        patch = overrides.get(node.cfg.node_id)
        if not patch:
            continue
        for key, value in patch.items():
            if key.startswith("main_seed."):
                node.main_seed[key.split(".", 1)[1]] = value
            elif key.startswith("side_seed."):
                node.side_seed[key.split(".", 1)[1]] = value
            elif hasattr(node.cfg.signal, key):
                setattr(node.cfg.signal, key, value)
            elif hasattr(node.cfg, key):
                setattr(node.cfg, key, value)
    return out


def _sat_flow_per_lane(seed: dict[str, Any]) -> float:
    sat = DEFAULT_SAT_FLOW_PER_LANE
    sat *= (1.0 - 0.5 * float(seed.get("truck_factor", 0.05)))
    return max(sat, 0.20)


def _build_link_state(name: str, seed: dict[str, Any], dt_s: float) -> LinkState:
    sat_flow_per_lane = _sat_flow_per_lane(seed)
    params = {
        "aadt": float(seed["aadt"]),
        "lanes": int(seed["lanes"]),
        "v_free": float(seed["speed_mph"]) * 0.44704,
        "length": float(seed["length_m"]),
        "sat_flow_per_lane": sat_flow_per_lane,
        "sat_flow": sat_flow_per_lane * int(seed["lanes"]),
        "green_frac": 0.50,
        "K": float(seed.get("k_factor", DEFAULT_K_FACTOR)),
        "D": float(seed.get("directional_factor", DEFAULT_DIRECTIONAL_FACTOR)),
        "turning_proportions": [[1.0]],
        "downstream_links": 1,
        "queue_storage_veh": DEFAULT_RHO_JAM * DEFAULT_CELL_LENGTH_M * int(seed["lanes"]),
    }
    engine, geometry, fd = build_ctm(params, dt_s)
    return LinkState(name=name, params=params, engine=engine, geometry=geometry, fd=fd)


def _build_links_and_queues(cfg: NetworkConfig):
    main_links = []
    side_links = []
    main_queues = []
    side_queues = []
    for idx, node in enumerate(cfg.intersections):
        main_link = _build_link_state(f"main_{idx}_{node.cfg.node_id}", node.main_seed, cfg.dt_s)
        side_link = _build_link_state(f"side_{idx}_{node.cfg.node_id}", node.side_seed, cfg.dt_s)
        main_links.append(main_link)
        side_links.append(side_link)
        main_queues.append(StopLineQueue(main_link.params["sat_flow"], storage_veh=main_link.params["queue_storage_veh"]))
        side_queues.append(StopLineQueue(side_link.params["sat_flow"], storage_veh=side_link.params["queue_storage_veh"]))
    return main_links, side_links, main_queues, side_queues


def _arrival_series(aadt: float, horizon_s: float, dt_s: float, K: float, D: float, burst_prob: float, mean_burst_size: float, intra_burst_headway_s: float, seed: int) -> np.ndarray:
    return PlatoonArrivalGenerator(
        horizon_s=horizon_s,
        dt=dt_s,
        base_rate_veh_per_s=aadt_to_peak_rate(aadt, K=K, D=D),
        burst_prob=burst_prob,
        mean_burst_size=mean_burst_size,
        intra_burst_headway_s=intra_burst_headway_s,
        seed=seed,
    ).generate()

#Adding more sophisticated spillback reporting support.
def approach_is_spilling(link, queue):
    return bool(queue.spillback_active or link.tail_blocked())

def approach_receiving_scale(link, queue):
    if queue.spillback_active or link.tail_blocked():
        return 0.0
    return 1.0

#The meat of the module.    
def simulate_network(cfg: NetworkConfig, seed: int = 42) -> dict[str, Any]:
    steps = int(cfg.horizon_s / cfg.dt_s)
    main_links, side_links, main_queues, side_queues = _build_links_and_queues(cfg)

    main_entry = _arrival_series(
        aadt=cfg.intersections[0].main_seed["aadt"],
        horizon_s=cfg.horizon_s,
        dt_s=cfg.dt_s,
        K=cfg.intersections[0].main_seed.get("k_factor", cfg.peak_k_factor),
        D=cfg.intersections[0].main_seed.get("directional_factor", cfg.default_directional_factor),
        burst_prob=cfg.burst_prob,
        mean_burst_size=cfg.mean_burst_size,
        intra_burst_headway_s=cfg.intra_burst_headway_s,
        seed=seed,
    )
    side_entries = [
        _arrival_series(
            aadt=node.side_seed["aadt"],
            horizon_s=cfg.horizon_s,
            dt_s=cfg.dt_s,
            K=node.side_seed.get("k_factor", cfg.peak_k_factor),
            D=node.side_seed.get("directional_factor", cfg.default_directional_factor),
            burst_prob=max(cfg.burst_prob * 0.8, 0.01),
            mean_burst_size=max(cfg.mean_burst_size - 1.0, 2.0),
            intra_burst_headway_s=cfg.intra_burst_headway_s,
            seed=seed + 100 + i,
        )
        for i, node in enumerate(cfg.intersections)
    ]

    timestamps = pd.date_range("2026-01-01 07:00:00", periods=steps, freq=f"{int(cfg.dt_s)}s")
    records: list[dict[str, Any]] = []
    corridor_queue = np.zeros(steps)
    corridor_tt = np.zeros(steps)
    spill_flags = np.zeros(steps)
    mainline_spill_flags = np.zeros(steps)
    side_spill_flags = np.zeros(steps)
    
    mainline_spill_extent = np.zeros(steps)
    side_spill_extent = np.zeros(steps)
    network_spill_extent = np.zeros(steps)
    entered_hist = np.zeros(steps)
    exited_hist = np.zeros(steps)

    upstream_main = np.zeros(len(cfg.intersections), dtype=float)
    upstream_side = np.zeros(len(cfg.intersections), dtype=float)
    
    carried_main_inflow = np.zeros(len(cfg.intersections), dtype=float)
    for k in range(steps):
        t_s = k * cfg.dt_s

        upstream_main[:] = carried_main_inflow
        upstream_side[:] = 0.0

        upstream_main[0] += float(main_entry[k])
        entered_hist[k] += float(main_entry[k]) * cfg.dt_s

        for i in range(len(cfg.intersections)):
            upstream_side[i] = float(side_entries[i][k])
            entered_hist[k] += upstream_side[i] * cfg.dt_s
        
        # Step 1: corridor links discharge into explicit stop-line queues.
        for i, node in enumerate(cfg.intersections):
            green_main, green_side = node.cfg.signal.green_fractions(t_s)
            main_rcv = main_queues[i].receiving_capacity(green_main, cfg.dt_s)
            side_rcv = side_queues[i].receiving_capacity(green_side, cfg.dt_s)
            main_flux = main_links[i].advance(upstream_main[i], main_rcv)
            side_flux = side_links[i].advance(upstream_side[i], side_rcv)
            main_queues[i].update(main_flux[-1], green_main, cfg.dt_s)
            side_queues[i].update(side_flux[-1], green_side, cfg.dt_s)

        # Step 2: node solver routes discharged queue vehicles subject to downstream receiving.
        next_main_inflow = np.zeros(len(cfg.intersections), dtype=float)
        for i, node in enumerate(cfg.intersections):
            q_send = np.array([
                main_queues[i].departures_last,
                side_queues[i].departures_last,
            ], dtype=float)
            next_main_recv = 1e6 if i == len(cfg.intersections) - 1 else main_links[i + 1].receiving_head()
            side_exit_recv = 1e6
            #Adding more sophisticated metrics.
            if i < len(cfg.intersections) - 1:
                next_main_scale = approach_receiving_scale(main_links[i + 1], main_queues[i + 1])
            else:
                next_main_scale = 1.0

            side_scale = approach_receiving_scale(side_links[i], side_queues[i])

            q_recv = np.array([
                next_main_recv * next_main_scale,
                side_exit_recv * side_scale,
            ], dtype=float)
            
            
            flows = godunov_node_solver(q_send, q_recv, turning_proportions=cfg.turning_proportions)
            to_main = float(flows[:, 0].sum())
            to_side = float(flows[:, 1].sum())
            if i < len(cfg.intersections) - 1:
                next_main_inflow[i + 1] += to_main
            else:
                exited_hist[k] += to_main * cfg.dt_s
            exited_hist[k] += to_side * cfg.dt_s

            records.append(
                {
                    "timestamp": timestamps[k],
                    "kind": "node",
                    "node_id": node.cfg.node_id,
                    "node_name": node.cfg.name,
                    "step": k,
                    "main_queue_depart_veh_s": q_send[0],
                    "side_queue_depart_veh_s": q_send[1],
                    "to_main_flow": to_main,
                    "to_side_flow": to_side,
                }
            )

        # Step 3: store next-step propagated mainline inflows.

        carried_main_inflow = next_main_inflow.copy()

        step_main_queue = 0.0
        
        mainline_spill_count = 0
        side_spill_count = 0
        for i, node in enumerate(cfg.intersections):
            main_q = float(main_queues[i].q)
            side_q = float(side_queues[i].q)

            step_main_queue += main_q + side_q
            
            mainline_spill = int(
                main_queues[i].spillback_active
                or main_links[i].tail_blocked()
            )

            side_spill = int(
                side_queues[i].spillback_active
                or side_links[i].tail_blocked()
            )

            network_spill = int(mainline_spill or side_spill)
            
            # store flags
            mainline_spill_flags[k] = max(mainline_spill_flags[k], mainline_spill)
            side_spill_flags[k] = max(side_spill_flags[k], side_spill)
            spill_flags[k] = max(spill_flags[k], network_spill)   
            mainline_spill_count += mainline_spill
            side_spill_count += side_spill
            records.append(
                {
                    "timestamp": timestamps[k],
                    "kind": "link",
                    "node_id": node.cfg.node_id,
                    "node_name": node.cfg.name,
                    "step": k,
                    "main_queue_veh": main_q,
                    "side_queue_veh": side_q,
                    "main_total_veh": main_links[i].total_veh(),
                    "side_total_veh": side_links[i].total_veh(),
                    "main_tt_s": main_links[i].travel_time_s(),
                    "side_tt_s": side_links[i].travel_time_s(),
                    "spillback_flag": network_spill,
                    "mainline_spillback_flag": mainline_spill,
                    "side_spillback_flag": side_spill,
                    "main_queue_storage_veh": main_queues[i].storage_veh,
                    "side_queue_storage_veh": side_queues[i].storage_veh,
                }
            )
        
        main_q_spill = sum(int(q.spillback_active) for q in main_queues)
        side_q_spill = sum(int(q.spillback_active) for q in side_queues)
        main_tail_spill = sum(int(link.tail_blocked()) for link in main_links)
        side_tail_spill = sum(int(link.tail_blocked()) for link in side_links)
        
        n_nodes = len(cfg.intersections)

        mainline_spill_extent[k] = mainline_spill_count / max(n_nodes, 1)
        side_spill_extent[k] = side_spill_count / max(n_nodes, 1)
        network_spill_extent[k] = (mainline_spill_count + side_spill_count) / max(2 * n_nodes, 1)
        
        corridor_queue[k] = step_main_queue
        corridor_tt[k] = sum(link.travel_time_s() for link in main_links)

    ts_df = pd.DataFrame(records)
    link_only = ts_df[ts_df["kind"] == "link"].copy()
    tt = travel_time_stats(corridor_tt)
    total_entered_veh = float(np.sum(entered_hist))
    total_exited_veh = float(np.sum(exited_hist))
    summary = {
        "mean_queue": float(compute_mean_queue(corridor_queue)),
        "std_queue": float(np.std(corridor_queue, ddof=0)),
        "mean_delay": float(compute_mean_delay(corridor_queue, total_exited_veh, cfg.horizon_s)),
        "p50_tt": float(tt["p50"]),
        "p90_tt": float(tt["p90"]),
        "spillback_freq": float(spillback_frequency(spill_flags)),
        "mainline_spillback_freq": float(np.mean(mainline_spill_flags)),
        "side_spillback_freq": float(np.mean(side_spill_flags)),
        "mainline_spill_extent": float(np.mean(mainline_spill_extent)),
        "side_spill_extent": float(np.mean(side_spill_extent)),
        "network_spill_extent": float(np.mean(network_spill_extent)),
        "total_entered_veh": total_entered_veh,
        "total_exited_veh": total_exited_veh,
        "horizon_s": float(cfg.horizon_s),
    }
    return {
        "summary": summary,
        "timeseries": ts_df,
        "link_timeseries": link_only,
        "corridor_queue": corridor_queue,
        "corridor_tt": corridor_tt,
        "spillback_flags": spill_flags,
        "network_config": cfg,
    }


def monte_carlo_network(cfg: NetworkConfig, runs: int = 20, seed0: int = 42, confidence: float = 0.95) -> dict[str, Any]:
    rows = []
    for r in range(runs):
        out = simulate_network(cfg, seed=seed0 + r)
        row = dict(out["summary"])
        row["run"] = r
        rows.append(row)
    df = pd.DataFrame(rows)

    queue_ci = mean_confidence_interval(df["mean_queue"].to_numpy(), confidence=confidence)
    delay_ci = mean_confidence_interval(df["mean_delay"].to_numpy(), confidence=confidence)
    p50_ci = mean_confidence_interval(df["p50_tt"].to_numpy(), confidence=confidence)
    p90_ci = mean_confidence_interval(df["p90_tt"].to_numpy(), confidence=confidence)
    spill_ci = mean_confidence_interval(df["spillback_freq"].to_numpy(), confidence=confidence)

    return {
        "mean_queue": queue_ci["mean"],
        "std_queue": float(df["mean_queue"].std(ddof=0) if len(df) > 1 else 0.0),
        "mean_queue_ci_low": queue_ci["lower"],
        "mean_queue_ci_high": queue_ci["upper"],
        "mean_delay": delay_ci["mean"],
        "std_delay": float(df["mean_delay"].std(ddof=0) if len(df) > 1 else 0.0),
        "mean_delay_ci_low": delay_ci["lower"],
        "mean_delay_ci_high": delay_ci["upper"],
        "p50_tt": p50_ci["mean"],
        "p50_tt_ci_low": p50_ci["lower"],
        "p50_tt_ci_high": p50_ci["upper"],
        "p90_tt": p90_ci["mean"],
        "p90_tt_ci_low": p90_ci["lower"],
        "p90_tt_ci_high": p90_ci["upper"],
        "spillback_freq": spill_ci["mean"],
        "spillback_ci_low": spill_ci["lower"],
        "spillback_ci_high": spill_ci["upper"],
        "mainline_spillback_freq": df["mainline_spillback_freq"].mean(),
        "side_spillback_freq": df["side_spillback_freq"].mean(),
        "mainline_spill_extent": float(df["mainline_spill_extent"].mean()),
        "side_spill_extent": float(df["side_spill_extent"].mean()),
        "network_spill_extent": float(df["network_spill_extent"].mean()),
        "total_entered_veh": float(df["total_entered_veh"].mean()),
        "total_exited_veh": float(df["total_exited_veh"].mean()),
        "ci_level": float(confidence),
        "sample_size": int(len(df)),
        "runs": df,
    }


def plot_scenario_report(df: pd.DataFrame, output_prefix: str = "report_network") -> None:
    plt.figure()
    plt.bar(df["scenario"], df["mean_queue"])
    plt.ylabel("Mean Queue")
    plt.title("Scenario Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_queue.png", dpi=150)
    plt.close()

    plt.figure()
    plt.bar(df["scenario"], df["p90_tt"])
    plt.ylabel("P90 Travel Time")
    plt.title("Reliability Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_p90.png", dpi=150)
    plt.close()

    plt.figure()
    plt.bar(df["scenario"], df["spillback_freq"])
    plt.ylabel("Spillback Frequency")
    plt.title("Spillback Risk")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_spillback.png", dpi=150)
    plt.close()


def write_network_metadata(cfg: NetworkConfig, out_csv: str = "network_metadata.csv") -> pd.DataFrame:
    rows = []
    for node in cfg.intersections:
        rows.append(
            {
                "node_id": node.cfg.node_id,
                "name": node.cfg.name,
                "main_road_id": node.main_seed.get("road_id"),
                "main_road_name": node.main_seed.get("road_name"),
                "main_county": node.main_seed.get("county"),
                "main_aadt": node.main_seed.get("aadt"),
                "main_lanes": node.main_seed.get("lanes"),
                "main_speed_mph": node.main_seed.get("speed_mph"),
                "side_road_id": node.side_seed.get("road_id"),
                "side_road_name": node.side_seed.get("road_name"),
                "side_county": node.side_seed.get("county"),
                "side_aadt": node.side_seed.get("aadt"),
                "side_lanes": node.side_seed.get("lanes"),
                "side_speed_mph": node.side_seed.get("speed_mph"),
                "cycle_s": node.cfg.signal.cycle_s,
                "green_main_s": node.cfg.signal.green_main_s,
                "green_side_s": node.cfg.signal.green_side_s,
                "offset_s": node.cfg.signal.offset_s,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    cfg = build_default_config()
    out = simulate_network(cfg, seed=42)
    print(pd.DataFrame([out["summary"]]))
