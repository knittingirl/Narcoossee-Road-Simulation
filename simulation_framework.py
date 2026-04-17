import numpy as np

from geometry import CorridorGeometry
from fundamental_diagram import FundamentalDiagram
from ctm_engine import CTMEngine
from queue_module import StopLineQueue
from arrival_generator import PlatoonArrivalGenerator
from postprocess import summarize_run, aggregate_monte_carlo

#These are set as reasonable values.

DEFAULT_CELL_LENGTH_M = 100.0
DEFAULT_RHO_CRIT = 0.03
DEFAULT_RHO_JAM = 0.15

#Computes peak hour flow, includes default values just in case.
def aadt_to_peak_rate(aadt, K=0.1, D=0.6):
    return (aadt * K * D) / 3600.0


def build_ctm(params, dt):
    n_cells = max(1, int(params["length"] / DEFAULT_CELL_LENGTH_M))

    geometry = CorridorGeometry(
        n_cells=n_cells,
        cell_length_m=DEFAULT_CELL_LENGTH_M,
        lanes_per_cell=params["lanes"],
    )

    fd = FundamentalDiagram(
        v_free=params["v_free"],
        rho_crit=DEFAULT_RHO_CRIT,
        rho_jam=DEFAULT_RHO_JAM,
        lanes=params["lanes"],
    )

    engine = CTMEngine(geometry, fd, dt)
    engine.initialize_density(np.zeros(n_cells))

    return engine, geometry, fd


def _total_saturation_flow(params):
    if "sat_flow_per_lane" in params:
        return params["sat_flow_per_lane"] * params["lanes"]
    return params["sat_flow"]


def _queue_storage_veh(params, geometry):
    if "queue_storage_veh" in params:
        return float(params["queue_storage_veh"])
    return float(geometry.cells[-1].storage_veh)


def run_simulation(params, horizon=3600, dt=1.0, seed=None):
    base_rate = aadt_to_peak_rate(
        params["aadt"],
        params.get("K", 0.088),
        params.get("D", 0.60),
    )

    arrivals = PlatoonArrivalGenerator(
        horizon_s=horizon,
        dt=dt,
        base_rate_veh_per_s=base_rate,
        seed=seed,
    ).generate()

    engine, geometry, fd = build_ctm(params, dt)
    total_sat_flow = _total_saturation_flow(params)
    queue_storage = _queue_storage_veh(params, geometry)
    queue = StopLineQueue(total_sat_flow, storage_veh=queue_storage)

    queue_series = []
    density_series = []
    travel_times = []

    # Cumulative-count style travel-time tracking to avoid truncating fractional flows.
    upstream_entry_times = []

    time = 0.0

    for inflow in arrivals:
        green_fraction = params["green_frac"]

        # The stop-line queue is the actual downstream bottleneck.
        # The corridor can send vehicles into the queue only if the queue has room now
        # or will free room through departures during this step.
        queue_receiving_capacity = queue.receiving_capacity(green_fraction, dt)

        flux = engine.step(
            external_inflow=inflow,
            downstream_capacity=queue_receiving_capacity,
        )

        arrivals_to_queue = flux[-1]
        queue.update(arrivals_to_queue, green_fraction, dt)

        # Fractional FIFO travel-time tracking using cumulative counts.
        entered_this_step = inflow * dt
        if entered_this_step > 0:
            upstream_entry_times.append([time, entered_this_step])

        exited_this_step = queue.departures_last
        remaining_to_match = exited_this_step
        while remaining_to_match > 1e-9 and upstream_entry_times:
            entry_time, pending_count = upstream_entry_times[0]
            matched = min(pending_count, remaining_to_match)
            if matched > 0:
                travel_times.extend([time - entry_time] * int(round(matched)))
            pending_count -= matched
            remaining_to_match -= matched
            if pending_count <= 1e-9:
                upstream_entry_times.pop(0)
            else:
                upstream_entry_times[0][1] = pending_count

        queue_series.append(queue.q)
        density_series.append(engine.rho.copy())
        time += dt

    travel_times = np.array(travel_times) if travel_times else np.array([0.0])

    # Spillback occurs when the stop-line queue hits its physical storage limit,
    # or when the last CTM cell is driven above critical density by downstream blockage.
    spillback_flag = queue.spillback_active or bool(np.any(engine.rho[-1:] >= fd.rho_crit))

    return {
        "queue": np.array(queue_series),
        "density": np.array(density_series),
        "travel_times": travel_times,
        "spillback": spillback_flag,
        "queue_storage_veh": queue_storage,
        "sat_flow_total": total_sat_flow,
    }


def monte_carlo(params, runs=50):
    run_summaries = []

    for i in range(runs):
        out = run_simulation(params, seed=i)
        run_summaries.append(summarize_run(out))

    return aggregate_monte_carlo(run_summaries)
