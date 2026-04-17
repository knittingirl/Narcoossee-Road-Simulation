import numpy as np
import pandas as pd


def compute_mean_queue(queue_time_series):
    arr = np.asarray(queue_time_series, dtype=float)
    return float(arr.mean()) if arr.size else 0.0



def compute_mean_delay(queue_time_series, total_exited_veh, horizon_s):
    """
    Approximate mean delay (s/veh) from the stop-line queue metric using Little's law.

    Delay ~= average queue / average departure rate.

    """
    mean_queue = compute_mean_queue(queue_time_series)
    horizon_s = float(horizon_s)
    total_exited_veh = float(total_exited_veh)
    if horizon_s <= 0.0 or total_exited_veh <= 1e-9:
        return 0.0
    avg_departure_rate = total_exited_veh / horizon_s
    if avg_departure_rate <= 1e-9:
        return 0.0
    return float(mean_queue / avg_departure_rate)



def travel_time_stats(travel_times_s):
    arr = np.asarray(travel_times_s, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }



def spillback_frequency(spillback_flags):
    arr = np.asarray(spillback_flags, dtype=float)
    return float(arr.mean()) if arr.size else 0.0



def mean_confidence_interval(values, confidence=0.95):
    """
    Normal-approximation confidence interval for the sample mean.

    Returns a dict with mean, lower, upper, half_width, and sample_size.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "half_width": 0.0,
            "sample_size": 0,
        }

    mean = float(arr.mean())
    if n == 1:
        return {
            "mean": mean,
            "lower": mean,
            "upper": mean,
            "half_width": 0.0,
            "sample_size": 1,
        }

    z_map = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_map.get(float(confidence), 1.959963984540054)
    std = float(arr.std(ddof=1))
    half_width = z * std / np.sqrt(n)
    return {
        "mean": mean,
        "lower": float(mean - half_width),
        "upper": float(mean + half_width),
        "half_width": float(half_width),
        "sample_size": n,
    }



def summarize_run(sim_output):
    """Summarize a single simulation output dict from run_simulation()."""
    tt = travel_time_stats(sim_output.get("travel_times", []))
    total_exited_veh = float(sim_output.get("total_exited_veh", 0.0))
    horizon_s = float(sim_output.get("horizon_s", 0.0))
    queue = sim_output.get("queue", [])
    return {
        "mean_queue": compute_mean_queue(queue),
        "mean_delay": compute_mean_delay(queue, total_exited_veh, horizon_s),
        "p50_tt": tt["p50"],
        "p90_tt": tt["p90"],
        "spillback": 1.0 if sim_output.get("spillback", False) else 0.0,
    }



def aggregate_monte_carlo(run_summaries, confidence=0.95):
    """Aggregate per-run summaries into the report table format."""
    df = pd.DataFrame(run_summaries)
    if df.empty:
        return {
            "mean_queue": 0.0,
            "std_queue": 0.0,
            "mean_delay": 0.0,
            "std_delay": 0.0,
            "p50_tt": 0.0,
            "p90_tt": 0.0,
            "spillback_freq": 0.0,
            "ci_level": confidence,
        }

    queue_ci = mean_confidence_interval(df["mean_queue"].to_numpy(), confidence=confidence)
    delay_ci = mean_confidence_interval(df["mean_delay"].to_numpy(), confidence=confidence)
    p50_ci = mean_confidence_interval(df["p50_tt"].to_numpy(), confidence=confidence)
    p90_ci = mean_confidence_interval(df["p90_tt"].to_numpy(), confidence=confidence)
    spill_ci = mean_confidence_interval(df["spillback"].to_numpy(), confidence=confidence)

    return {
        "mean_queue": queue_ci["mean"],
        "std_queue": float(df["mean_queue"].std(ddof=0)),
        "mean_queue_ci_low": queue_ci["lower"],
        "mean_queue_ci_high": queue_ci["upper"],
        "mean_delay": delay_ci["mean"],
        "std_delay": float(df["mean_delay"].std(ddof=0)),
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
        "ci_level": float(confidence),
        "sample_size": int(len(df)),
    }
