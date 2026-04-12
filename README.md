# Narcoossee-Road-Simulation
This is for the COT6571 course project. 
# geometry.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Cell:
    length_m: float
    lanes: int
    storage_veh: float

class CorridorGeometry:
    def __init__(self, n_cells: int, cell_length_m: float, lanes_per_cell: int):
        self.n_cells = n_cells
        self.cell_length_m = cell_length_m
        self.lanes_per_cell = lanes_per_cell
        self.cells = [Cell(length_m=cell_length_m, lanes=lanes_per_cell, storage_veh=self._default_storage())
                      for _ in range(n_cells)]

    def _default_storage(self):
        # rough storage capacity: jam density ~ 150 veh/km/lane
        jam_density_veh_per_m = 150 / 1000.0
        return jam_density_veh_per_m * self.cell_length_m * self.lanes_per_cell

    def total_length(self):
        return self.n_cells * self.cell_length_m

    def as_dict(self):
        return {"n_cells": self.n_cells, "cell_length_m": self.cell_length_m, "lanes_per_cell": self.lanes_per_cell}

# fundamental_diagram.py
import numpy as np
from dataclasses import dataclass

@dataclass
class FundamentalDiagram:
    v_free: float        # free-flow speed (m/s)
    rho_crit: float      # critical density (veh/m)
    rho_jam: float       # jam density (veh/m)
    lanes: int = 1

    def flow_capacity(self):
        return self.v_free * self.rho_crit * self.lanes

    def sending(self, rho):
        """
        Sending function S(rho) = min(v_free * rho, capacity)
        rho: density in veh/m (can be array)
        """
        return np.minimum(self.v_free * rho * self.lanes, self.flow_capacity())

    def receiving(self, rho):
        """
        Receiving function R(rho) = w * (rho_jam - rho)
        where w is backward wave speed computed from FD parameters
        """
        # compute backward wave speed w from triangular FD: w = q_max / (rho_jam - rho_crit)
        q_max = self.flow_capacity()
        w = q_max / (self.rho_jam - self.rho_crit + 1e-9)
        return w * (self.rho_jam - rho) * self.lanes

# ctm_engine.py
import numpy as np

class CTMEngine:
    def __init__(self, geometry, fd: 'FundamentalDiagram', dt: float):
        self.geo = geometry
        self.fd = fd
        self.dt = dt
        self.n = geometry.n_cells
        # densities in veh/m per lane aggregated across lanes
        self.rho = np.zeros(self.n)  # veh/m (per lane)
        # convert storage to densities if needed
        self.cell_length = geometry.cell_length_m

    def initialize_density(self, rho_init):
        rho_init = np.asarray(rho_init)
        assert rho_init.shape[0] == self.n
        self.rho = rho_init

    def step(self, external_inflow=0.0, receiving_multiplier=None):
        """
        One CTM time step with conservative fluxes.
        external_inflow: vehicles entering at upstream boundary (veh/s)
        receiving_multiplier: optional array of multipliers (0-1) applied to receiving of each cell
        """
        S = self.fd.sending(self.rho)  # veh/s (per lane aggregated by lanes inside FD)
        R = self.fd.receiving(self.rho)
        if receiving_multiplier is not None:
            R = R * receiving_multiplier

        # fluxes between i -> i+1
        flux = np.zeros(self.n + 1)  # flux[0] is upstream boundary, flux[n] is outflow
        # upstream boundary limited by upstream sending and first cell receiving
        flux[0] = min(external_inflow, R[0])
        for i in range(self.n - 1):
            flux[i+1] = min(S[i], R[i+1])
        # outflow from last cell is its sending
        flux[self.n] = S[-1]

        # update densities conservatively: rho_new = rho + (dt/length) * (inflow - outflow)
        inflow = flux[:-1]
        outflow = flux[1:]
        delta = (self.dt / self.cell_length) * (inflow - outflow)
        self.rho = np.maximum(self.rho + delta, 0.0)
        return flux

# node_solver.py
import numpy as np

def godunov_node_solver(upstream_sending, downstream_receiving, turning_proportions=None):
    """
    Simple Godunov-style node solver for a single merge/diverge node.
    upstream_sending: array of sending capacities from upstream links (veh/s)
    downstream_receiving: array of receiving capacities for downstream links (veh/s)
    turning_proportions: matrix shape (n_up, n_down) with fractions summing to 1 per upstream link
    Returns flow matrix f[i,j] from upstream i to downstream j
    """
    n_up = upstream_sending.shape[0]
    n_down = downstream_receiving.shape[0]
    if turning_proportions is None:
        # default: all upstream goes to first downstream
        tp = np.zeros((n_up, n_down))
        tp[:, 0] = 1.0
    else:
        tp = turning_proportions

    # initial desired flows
    desired = (upstream_sending[:, None] * tp)
    # allocate respecting downstream receiving
    f = np.zeros_like(desired)
    remaining_R = downstream_receiving.copy()
    # simple proportional allocation
    for j in range(n_down):
        demand_to_j = desired[:, j]
        total_demand = demand_to_j.sum()
        if total_demand <= remaining_R[j] + 1e-9:
            f[:, j] = demand_to_j
            remaining_R[j] -= total_demand
        else:
            # scale demands proportionally
            if total_demand > 0:
                f[:, j] = demand_to_j * (remaining_R[j] / total_demand)
                remaining_R[j] = 0.0
    return f

# queue_module.py
import numpy as np

class StopLineQueue:
    def __init__(self, saturation_flow_veh_per_s: float):
        self.q = 0.0
        self.saturation_flow = saturation_flow_veh_per_s

    def update(self, arrivals, green_fraction, dt):
        """
        Discrete queue update at time resolution dt
        q(t+dt) = q(t) + arrivals*dt - departures
        departures limited by green_fraction * saturation_flow * dt
        arrivals: veh/s entering the queue during dt
        green_fraction: fraction of cycle that is green (0-1) during this dt
        """
        arrivals_veh = arrivals * dt
        max_departures = green_fraction * self.saturation_flow * dt
        departures = min(self.q + arrivals_veh, max_departures)
        self.q = max(self.q + arrivals_veh - departures, 0.0)
        return departures

# arrival_generator.py
import numpy as np

class PlatoonArrivalGenerator:
    def __init__(self, horizon_s, dt, base_rate_veh_per_s, burst_prob=0.1, mean_burst_size=5, intra_burst_headway_s=1.0, seed=None):
        self.horizon_s = horizon_s
        self.dt = dt
        self.steps = int(np.ceil(horizon_s / dt))
        self.base_rate = base_rate_veh_per_s
        self.burst_prob = burst_prob
        self.mean_burst_size = mean_burst_size
        self.intra_burst_headway = intra_burst_headway_s
        self.rng = np.random.default_rng(seed)

    def generate(self):
        """
        Returns arrivals array of length steps with arrival rate (veh/s) per dt interval.
        Mixes Poisson baseline with occasional bursts (platoons).
        """
        rates = np.full(self.steps, self.base_rate)
        # add time-varying modulation (example: simple peak in middle)
        t = np.arange(self.steps) * self.dt
        peak = 1.0 + 0.5 * np.exp(-((t - self.horizon_s/2)**2) / (2*(self.horizon_s/6)**2))
        rates *= peak

        # baseline Poisson arrivals converted to rate per second already represented by rates
        arrivals = np.zeros(self.steps)
        for i in range(self.steps):
            # baseline Poisson count in dt
            lam = rates[i] * self.dt
            baseline_count = self.rng.poisson(lam)
            arrivals[i] += baseline_count / self.dt  # convert back to veh/s for consistency

            # burst generation
            if self.rng.random() < self.burst_prob:
                burst_size = max(1, self.rng.poisson(self.mean_burst_size))
                # place burst vehicles across subsequent time steps according to intra-burst headway
                for k in range(burst_size):
                    arrival_time = i * self.dt + k * self.intra_burst_headway
                    idx = int(np.floor(arrival_time / self.dt))
                    if idx < self.steps:
                        arrivals[idx] += 1.0 / self.dt
        return arrivals

# postprocess.py
import numpy as np
import pandas as pd

def compute_mean_queue(queue_time_series):
    arr = np.asarray(queue_time_series)
    return arr.mean()

def travel_time_stats(travel_times_s):
    arr = np.asarray(travel_times_s)
    return {"mean": arr.mean(), "p50": np.percentile(arr, 50), "p90": np.percentile(arr, 90)}

def spillback_frequency(spillback_flags):
    return np.sum(spillback_flags)
