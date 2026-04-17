#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

from runner_network_flexible import (
    adjust_mainline_green,
    build_network_config_from_feeder_cosites,
    monte_carlo_network,
)
from network_data import FEEDER_INTERSECTIONS, CORRIDOR


def main() -> None:
    base = build_network_config_from_feeder_cosites(
        corridor=CORRIDOR,
        feeders=FEEDER_INTERSECTIONS,
    )

    green_deltas_s = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    runs = 20
    rows = []

    for delta_s in green_deltas_s:
        cfg = adjust_mainline_green(base, delta_s)
        res = monte_carlo_network(cfg, runs=runs)
        rows.append({
            "green_delta_s": delta_s,
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
    df.to_csv("runner_lines_green_results.csv", index=False)
    print(df)

    plt.figure()
    plt.plot(df["green_delta_s"], df["mean_queue"], marker="o")
    plt.xlabel("Mainline Green Adjustment (s)")
    plt.ylabel("Mean Queue (veh)")
    plt.title("Green Timing Sensitivity: Mean Queue")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_green_queue.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["green_delta_s"], df["mean_delay"], marker="o")
    plt.xlabel("Mainline Green Adjustment (s)")
    plt.ylabel("Mean Delay (s/veh)")
    plt.title("Green Timing Sensitivity: Mean Delay")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_green_delay.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["green_delta_s"], df["p90_tt"], marker="o")
    plt.xlabel("Mainline Green Adjustment (s)")
    plt.ylabel("P90 Travel Time (s)")
    plt.title("Green Timing Sensitivity: Reliability")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_green_p90.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["green_delta_s"], df["network_spill_extent"], marker="o")
    plt.xlabel("Mainline Green Adjustment (s)")
    plt.ylabel("Network Spill Extent")
    plt.title("Green Timing Sensitivity: Spillback Extent")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_green_spill_extent.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["green_delta_s"], df["total_exited_veh"], marker="o")
    plt.xlabel("Mainline Green Adjustment (s)")
    plt.ylabel("Total Exited Vehicles")
    plt.title("Green Timing Sensitivity: Throughput")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_green_throughput.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
