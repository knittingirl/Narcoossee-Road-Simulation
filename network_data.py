from runner_network_flexible import FeederIntersectionSpec, SignalPlan, CorridorSpec

#Precise signal timing cannot be obtained based on public information, so these are some best guesses.
small_signal = SignalPlan(cycle_s=100, green_main_s=60, green_side_s=25)
large_signal = SignalPlan(cycle_s=120, green_main_s=70, green_side_s=35)
# The basic idea is that this allows us to put in any number of cosite IDs, which are identifiers for the side roads in this model.

#We filled these in by collecting IDs from the interactive FDOT map here: https://gis-fdot.opendata.arcgis.com/datasets/fdot::annual-average-daily-traffic-tda/explore?location=28.272234%2C-81.235165%2C13

FEEDER_INTERSECTIONS = [
    FeederIntersectionSpec(cosite="920155", node_id="node_1", name="Hickory Tree Road East", signal=large_signal),
    FeederIntersectionSpec(cosite="920255", node_id="node_2", name="Hickory Tree Road West", signal=large_signal),
    FeederIntersectionSpec(cosite="752264", node_id="node_3", name="SR417 Ramp West", signal=large_signal),
    FeederIntersectionSpec(cosite="752263", node_id="node_4", name="SR417 Ramp East", signal=large_signal),   
]

# Main corridor road remains a single shared specification.
CORRIDOR = CorridorSpec(road_id="92050000", county="Osceola")