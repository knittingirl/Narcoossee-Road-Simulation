
import pandas as pd
import matplotlib.pyplot as plt

from runner_network_flexible import (
    build_network_config_from_feeder_cosites,
    monte_carlo_network,
    scale_all_mainline_demand,
)
from network_data import FEEDER_INTERSECTIONS, CORRIDOR


def main() -> None:
    base = build_network_config_from_feeder_cosites(
        corridor=CORRIDOR,
        feeders=FEEDER_INTERSECTIONS,
    )

    demand_scales = [0.70, 0.85, 1.00, 1.15, 1.30, 1.45, 1.60]
    runs = 20
    rows = []

    for scale in demand_scales:
        cfg = scale_all_mainline_demand(base, scale)
        res = monte_carlo_network(cfg, runs=runs)
        rows.append({
            "demand_scale": scale,
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
    df.to_csv("runner_lines_demand_results.csv", index=False)
    print(df)

    plt.figure()
    plt.plot(df["demand_scale"], df["mean_queue"], marker="o")
    plt.xlabel("Mainline Demand Scale")
    plt.ylabel("Mean Queue (veh)")
    plt.title("Demand Sensitivity: Mean Queue")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_demand_queue.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["demand_scale"], df["mean_delay"], marker="o")
    plt.xlabel("Mainline Demand Scale")
    plt.ylabel("Mean Delay (s/veh)")
    plt.title("Demand Sensitivity: Mean Delay")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_demand_delay.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["demand_scale"], df["p90_tt"], marker="o")
    plt.xlabel("Mainline Demand Scale")
    plt.ylabel("P90 Travel Time (s)")
    plt.title("Demand Sensitivity: Reliability")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_demand_p90.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["demand_scale"], df["network_spill_extent"], marker="o")
    plt.xlabel("Mainline Demand Scale")
    plt.ylabel("Network Spill Extent")
    plt.title("Demand Sensitivity: Spillback Extent")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_demand_spill_extent.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["demand_scale"], df["total_exited_veh"], marker="o")
    plt.xlabel("Mainline Demand Scale")
    plt.ylabel("Total Exited Vehicles")
    plt.title("Demand Sensitivity: Throughput")
    plt.grid()
    plt.tight_layout()
    plt.savefig("runner_lines_demand_throughput.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
