#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

from runner_network_flexible import (
    adjust_mainline_green,
    build_network_config_from_feeder_cosites,
    monte_carlo_network,
    scale_all_mainline_demand,
)
from network_data import FEEDER_INTERSECTIONS, CORRIDOR


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str) -> None:
    plt.figure()
    for delta_s in sorted(df["green_delta_s"].unique()):
        subset = df[df["green_delta_s"] == delta_s].sort_values("demand_scale")
        label = f"Green {delta_s:+.0f}s"
        plt.plot(subset["demand_scale"], subset[metric], marker="o", label=label)
    plt.xlabel("Mainline Demand Scale")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend(title="Green Change")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main() -> None:
    base = build_network_config_from_feeder_cosites(
        corridor=CORRIDOR,
        feeders=FEEDER_INTERSECTIONS,
    )

    demand_scales = [0.70, 0.85, 1.00, 1.15, 1.30, 1.45, 1.60]
    green_deltas_s = [-20.0, -10.0, 0.0, 10.0, 20.0]
    runs = 20
    rows = []

    for delta_s in green_deltas_s:
        green_cfg = adjust_mainline_green(base, delta_s)
        for scale in demand_scales:
            cfg = scale_all_mainline_demand(green_cfg, scale)
            res = monte_carlo_network(cfg, runs=runs)
            rows.append({
                "green_delta_s": delta_s,
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
                "mean_queue_ci_low": res.get("mean_queue_ci_low"),
                "mean_queue_ci_high": res.get("mean_queue_ci_high"),
                "mean_delay_ci_low": res.get("mean_delay_ci_low"),
                "mean_delay_ci_high": res.get("mean_delay_ci_high"),
                "p90_tt_ci_low": res.get("p90_tt_ci_low"),
                "p90_tt_ci_high": res.get("p90_tt_ci_high"),
                "spillback_ci_low": res.get("spillback_ci_low"),
                "spillback_ci_high": res.get("spillback_ci_high"),
                "sample_size": res.get("sample_size"),
                "ci_level": res.get("ci_level"),
            })

    df = pd.DataFrame(rows).sort_values(["green_delta_s", "demand_scale"]).reset_index(drop=True)
    df.to_csv("runner_lines_green_vs_demand_results.csv", index=False)
    print(df)

    _plot_metric(
        df,
        metric="mean_queue",
        ylabel="Mean Queue (veh)",
        title="Green vs Demand: Mean Queue",
        filename="runner_lines_green_vs_demand_queue.png",
    )
    _plot_metric(
        df,
        metric="mean_delay",
        ylabel="Mean Delay (s/veh)",
        title="Green vs Demand: Mean Delay",
        filename="runner_lines_green_vs_demand_delay.png",
    )
    _plot_metric(
        df,
        metric="p90_tt",
        ylabel="P90 Travel Time (s)",
        title="Green vs Demand: Reliability",
        filename="runner_lines_green_vs_demand_p90.png",
    )
    _plot_metric(
        df,
        metric="network_spill_extent",
        ylabel="Network Spill Extent",
        title="Green vs Demand: Spillback Extent",
        filename="runner_lines_green_vs_demand_spill_extent.png",
    )
    _plot_metric(
        df,
        metric="total_exited_veh",
        ylabel="Total Exited Vehicles",
        title="Green vs Demand: Throughput",
        filename="runner_lines_green_vs_demand_throughput.png",
    )


if __name__ == "__main__":
    main()
