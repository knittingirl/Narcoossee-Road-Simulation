import pandas as pd
import matplotlib.pyplot as plt
from runner_network_flexible import (
    CorridorSpec,
    FeederIntersectionSpec,
    SignalPlan,
    add_lane_to_all_mainline_segments,
    adjust_mainline_green,
    build_network_config_from_feeder_cosites,
    monte_carlo_network,
    scale_all_mainline_demand,
)
from network_data import FEEDER_INTERSECTIONS, CORRIDOR


def main():
    if any(str(item.cosite).startswith("REPLACE_COSITE") for item in FEEDER_INTERSECTIONS):
        raise ValueError("Update FEEDER_INTERSECTIONS with real feeder-road COSITE values before running.")

    base = build_network_config_from_feeder_cosites(
    corridor=CORRIDOR,
    feeders=FEEDER_INTERSECTIONS
    )
    print("Feeders:", len(FEEDER_INTERSECTIONS))
    print("Built intersections:", len(base.intersections))
    scenarios = {
        "Baseline": base,
        "Add Lane + More Green": adjust_mainline_green(
        add_lane_to_all_mainline_segments(base, 1),
        10.0,
        ),
        "More Green": adjust_mainline_green(base, 10.0),
        "Add Lane": add_lane_to_all_mainline_segments(base, 1),
        "High Demand": scale_all_mainline_demand(base, 1.15),
        "Oversaturated": scale_all_mainline_demand(base, 1.25),
    }
    '''
        "More Green": adjust_mainline_green(base, 10.0),
        "Add Lane": add_lane_to_all_mainline_segments(base, 1),
        "High Demand": scale_all_mainline_demand(base, 1.15),
        "Oversaturated": scale_all_mainline_demand(base, 1.25),
    }
    '''
    results = []
    for name, cfg in scenarios.items():
        res = monte_carlo_network(cfg, runs=2)
        print(f"\n--- {name} ---")
        print(f"entered: {res['total_entered_veh']:.1f}")
        print(f"exited:  {res['total_exited_veh']:.1f}")
        print(f"mean_queue: {res['mean_queue']:.2f}")
        print(f"mean_delay: {res['mean_delay']:.2f}")
        print(f"p50_tt: {res['p50_tt']:.2f}")
        print(f"p90_tt: {res['p90_tt']:.2f}")
        print(f"spillback_freq: {res['spillback_freq']:.3f}")
        print(f"mainline_spillback: {res['mainline_spillback_freq']:.3f}")
        print(f"side_spillback:     {res['side_spillback_freq']:.3f}")
        print(f"mainline_spill_extent: {res['mainline_spill_extent']:.3f}")
        print(f"side_spill_extent:     {res['side_spill_extent']:.3f}")
        print(f"network_spill_extent:  {res['network_spill_extent']:.3f}")
        res["scenario"] = name
        results.append(res)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("report_results_network.csv", index=False)
    plt.figure(); plt.bar(df["scenario"], df["mean_queue"]); plt.ylabel("Mean Queue"); plt.title("Scenario Comparison"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig("report_network_queue.png"); plt.close()
    plt.figure(); plt.bar(df["scenario"], df["p90_tt"]); plt.ylabel("P90 Travel Time"); plt.title("Reliability Comparison"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig("report_network_p90.png"); plt.close()
    plt.figure(); plt.bar(df["scenario"], df["spillback_freq"]); plt.ylabel("Spillback Frequency"); plt.title("Spillback Risk"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig("report_network_spillback.png"); plt.close()

if __name__ == '__main__':
    main()
