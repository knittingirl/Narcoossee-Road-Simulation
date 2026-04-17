class StopLineQueue:
    def __init__(self, saturation_flow_veh_per_s: float, storage_veh: float | None = None):
        self.q = 0.0
        self.saturation_flow = saturation_flow_veh_per_s
        self.storage_veh = float(storage_veh) if storage_veh is not None else float('inf')
        self.departures_last = 0.0
        self.spillback_active = False

    def receiving_capacity(self, green_fraction: float, dt: float) -> float:
        free_storage = max(self.storage_veh - self.q, 0.0)
        departures_room = green_fraction * self.saturation_flow * dt
        return (free_storage + departures_room) / max(dt, 1e-9)

    def update(self, arrivals: float, green_fraction: float, dt: float):
        arrivals_veh = arrivals * dt
        max_departures = green_fraction * self.saturation_flow * dt
        departures = min(self.q + arrivals_veh, max_departures)
        self.q = min(max(self.q + arrivals_veh - departures, 0.0), self.storage_veh)
        self.departures_last = departures / max(dt, 1e-9)
        self.spillback_active = self.q >= self.storage_veh - 1e-9
        return departures
