#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

from runner_network_flexible import (
    add_lane_to_all_mainline_segments,
    build_network_config_from_feeder_cosites,
    monte_carlo_network,
)
from network_data import FEEDER_INTERSECTIONS, CORRIDOR


def main() -> None:
    base = build_network_config_from_feeder_cosites(
        corridor=CORRIDOR,
        feeders=FEEDER_INTERSECTIONS,
    )

    extra_lanes_values = [0, 1, 2, 3, 4]
    runs = 20
    rows = []

    for extra_lanes in extra_lanes_values:
        cfg = add_lane_to_all_mainline_segments(base, extra_lanes) if extra_lanes else base
        res = monte_carlo_network(cfg, runs=runs)
        rows.append({
            "extra_lanes": extra_lanes,
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
    df.to_csv("runner_lines_lanes_results.csv", index=False)
    print(df)

    plt.figure()
    plt.plot(df["extra_lanes"], df["mean_queue"], marker="o")
    plt.xlabel("Extra Mainline Lanes Added")
    plt.ylabel("Mean Queue (veh)")
    plt.title("Lane Sensitivity: Mean Queue")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_lanes_queue.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["extra_lanes"], df["mean_delay"], marker="o")
    plt.xlabel("Extra Mainline Lanes Added")
    plt.ylabel("Mean Delay (s/veh)")
    plt.title("Lane Sensitivity: Mean Delay")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_lanes_delay.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["extra_lanes"], df["p90_tt"], marker="o")
    plt.xlabel("Extra Mainline Lanes Added")
    plt.ylabel("P90 Travel Time (s)")
    plt.title("Lane Sensitivity: Reliability")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_lanes_p90.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["extra_lanes"], df["network_spill_extent"], marker="o")
    plt.xlabel("Extra Mainline Lanes Added")
    plt.ylabel("Network Spill Extent")
    plt.title("Lane Sensitivity: Spillback Extent")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_lanes_spill_extent.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["extra_lanes"], df["total_exited_veh"], marker="o")
    plt.xlabel("Extra Mainline Lanes Added")
    plt.ylabel("Total Exited Vehicles")
    plt.title("Lane Sensitivity: Throughput")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_lanes_throughput.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
