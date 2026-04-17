#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

from runner_network_flexible import (
    build_network_config_from_feeder_cosites,
    monte_carlo_network,
    clone_config,
)
from network_data import FEEDER_INTERSECTIONS, CORRIDOR


def adjust_mainline_speed(cfg, delta_mph):
    out = clone_config(cfg)
    for node in out.intersections:
        node.main_seed["speed_mph"] = max(
            5.0, float(node.main_seed["speed_mph"]) + delta_mph
        )
    return out


def main() -> None:
    base = build_network_config_from_feeder_cosites(
        corridor=CORRIDOR,
        feeders=FEEDER_INTERSECTIONS,
    )

    speed_deltas = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    runs = 20
    rows = []

    for delta in speed_deltas:
        cfg = adjust_mainline_speed(base, delta)
        res = monte_carlo_network(cfg, runs=runs)

        rows.append({
            "speed_delta_mph": delta,
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
    df.to_csv("runner_lines_speed_results.csv", index=False)
    print(df)

    # --- plots ---
    def plot(y, ylabel, title, fname):
        plt.figure()
        plt.plot(df["speed_delta_mph"], df[y], marker="o")
        plt.xlabel("Speed Change (mph)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    plot("mean_queue", "Mean Queue (veh)", "Speed Sensitivity: Mean Queue",
         "runner_lines_speed_queue.png")

    plot("mean_delay", "Mean Delay (s/veh)", "Speed Sensitivity: Mean Delay",
         "runner_lines_speed_delay.png")

    plot("p90_tt", "P90 Travel Time (s)", "Speed Sensitivity: Reliability",
         "runner_lines_speed_p90.png")

    plot("network_spill_extent", "Network Spill Extent",
         "Speed Sensitivity: Spillback",
         "runner_lines_speed_spill_extent.png")

    plot("total_exited_veh", "Total Exited Vehicles",
         "Speed Sensitivity: Throughput",
         "runner_lines_speed_throughput.png")


if __name__ == "__main__":
    main()